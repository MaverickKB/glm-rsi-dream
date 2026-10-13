"""Append-only candidate snapshots, with the latest line owning each id."""

import fcntl
import hashlib
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

STATUSES = {"pending", "accepted", "rejected", "absorbed", "staged", "learned"}
NOVELTIES = {"new", "conflicts", "known", "unmeasured"}


def candidate_id(source_ref: Any, text: Any) -> Any:
    return hashlib.sha256((source_ref + text).encode("utf-8")).hexdigest()[:16]


def validate_record(record: Any) -> Any:
    """Reject incomplete snapshots before they can become the latest state."""
    if not isinstance(record, dict):
        raise ValueError("candidate must be an object")
    source = record.get("source")
    if not isinstance(source, dict) or source.get("kind") not in {"prompt", "conversation", "manual"}:
        raise ValueError("source kind must be prompt, conversation or manual")
    source_ref = source.get("ref")
    if not isinstance(source_ref, str) or not source_ref.strip():
        raise ValueError("source ref must be nonempty text")
    for field in ("text", "prompt_evidence", "created_et"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(field + " must be nonempty text")
    if record.get("id") != candidate_id(source_ref, record["text"]):
        raise ValueError("candidate id does not match source ref and text")
    created = datetime.fromisoformat(record["created_et"])
    if created.tzinfo is None:
        raise ValueError("created_et must include a timezone")
    if record.get("status") not in STATUSES or record.get("novelty") not in NOVELTIES:
        raise ValueError("invalid status or novelty")
    if record["novelty"] == "unmeasured" and source["kind"] != "manual":
        raise ValueError("unmeasured novelty requires manual input")
    learned = record.get("learned")
    if record["status"] == "learned":
        if not isinstance(learned, dict) or not isinstance(learned.get("receipt"), str) or not learned["receipt"].strip():
            raise ValueError("learned items require an explicit provenance receipt")
    review = record.get("review")
    if review is not None:
        if not isinstance(review, dict) or type(review.get("safe")) is not bool or review.get("reviewer") != "glm":
            raise ValueError("review requires a GLM safety result")
        if not isinstance(review.get("reason"), str) or not review["reason"].strip():
            raise ValueError("review requires a reason")
    change = record.get("change")
    if change is not None:
        if not isinstance(change, dict) or change.get("kind") not in {"new", "duplicate", "update"}:
            raise ValueError("invalid change kind")
        if change.get("reviewer") != "glm" or not isinstance(change.get("reason"), str) or not change["reason"].strip():
            raise ValueError("change requires a GLM reason")
        links = change.get("targets")
        if not isinstance(links, list) or (change["kind"] != "new" and not links):
            raise ValueError("duplicate/update requires linked learned targets")
        if change["kind"] == "new" and links:
            raise ValueError("new items cannot replace learned targets")
        seen = set()
        for link in links:
            if not isinstance(link, dict) or not isinstance(link.get("id"), str):
                raise ValueError("invalid learned target")
            if link["id"] == record["id"] or link["id"] in seen:
                raise ValueError("update targets must be distinct older items")
            if not isinstance(link.get("text_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", link["text_sha256"]):
                raise ValueError("learned target requires a content hash")
            for field in ("statement", "receipt"):
                if not isinstance(link.get(field), str) or not link[field].strip():
                    raise ValueError("learned target requires " + field)
            seen.add(link["id"])
    prior = record.get("glm_prior")
    if not isinstance(prior, dict) or not isinstance(prior.get("answer"), str):
        raise ValueError("glm_prior must contain an answer string")
    if "agrees" not in prior or (prior["agrees"] is not None and type(prior["agrees"]) is not bool):
        raise ValueError("glm_prior.agrees must be bool or null")
    grade = record.get("grade")
    if "grade" not in record:
        raise ValueError("grade must be present, initially null")
    if grade is not None:
        if not isinstance(grade, dict) or grade.get("verdict") not in {"accept", "reject"}:
            raise ValueError("invalid grade verdict")
        if grade.get("route") not in {"weights", "pocket", "memory"} or grade.get("grader") != "glm":
            raise ValueError("invalid GLM grade route or grader")
        score: Any = grade.get("score")
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("grade score must be between zero and one")
        if not isinstance(grade.get("reason"), str) or not grade["reason"].strip():
            raise ValueError("grade reason must be nonempty text")
        pocket = grade.get("pocket")
        if "pocket" not in grade or (pocket is not None and not isinstance(pocket, str)):
            raise ValueError("grade pocket must be text or null")
        if grade["verdict"] == "accept" and grade["route"] == "pocket" and (pocket is None or not pocket.strip()):
            raise ValueError("pocket route requires a pocket name")
        if source["kind"] == "manual" and grade["verdict"] == "accept":
            if review is None or review["safe"] is not True or change is None:
                raise ValueError("manual acceptance requires safe GLM relationship review")
    timings = record.get("timings_ms")
    if not isinstance(timings, dict):
        raise ValueError("timings_ms must be an object")
    for stage in ("extract", "probe", "grade", "absorb"):
        elapsed: Any = timings.get(stage)
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("invalid timing for " + stage)


def append_record(path: Any, record: Any) -> Any:
    validate_record(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(line)
        handle.flush()
        # Completed snapshots survive an ordinary process exit or restart.
        os.fsync(handle.fileno())


def latest_records(path: Any) -> Any:
    path = Path(path)
    if not path.exists():
        return []
    latest = {}
    with path.open(encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                validate_record(record)
            except (ValueError, TypeError) as error:
                raise ValueError(f"invalid queue line {line_number}: {error}") from error
            latest[record["id"]] = record
    return list(latest.values())


def record_revision(record: Any) -> Any:
    raw = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def replace_records(path: Any, expected: Any, records: Any) -> Any:
    """Compare snapshots and append a validated batch under one queue lock."""
    lines = []
    identifiers = set()
    for record in records:
        validate_record(record)
        if record["id"] in identifiers:
            raise ValueError("Queue replacement ids must be distinct")
        identifiers.add(record["id"])
        lines.append(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    if identifiers != set(expected):
        raise ValueError("Every queue replacement requires an expected revision")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        current = {}
        for line in handle:
            if line.strip():
                record = json.loads(line)
                validate_record(record)
                current[record["id"]] = record
        for identifier, revision in expected.items():
            previous = current.get(identifier)
            actual = record_revision(previous) if previous is not None else None
            if actual != revision:
                raise ValueError("Queue revision changed: " + identifier)
        handle.seek(0, os.SEEK_END)
        handle.write("".join(lines))
        handle.flush()
        os.fsync(handle.fileno())


def list_candidates(path: Any, status: Any = None) -> Any:
    if status is not None and status not in STATUSES:
        raise ValueError("unknown queue status: " + str(status))
    records = latest_records(path)
    if status is None:
        return records
    matching = []
    for record in records:
        if record["status"] == status:
            matching.append(record)
    return matching


def format_table(records: Any) -> Any:
    lines = ["ID                STATUS    NOVELTY    VERDICT         ROUTE    SOURCE        CHANGE    TARGETS           TEXT"]
    for record in records:
        route = "-"
        verdict = "-"
        decision = record["grade"]
        if decision is None and "proposal" in record:
            decision = record["proposal"]["grade"]
            verdict = "propose:" + decision["verdict"]
        elif decision is not None:
            verdict = decision["verdict"]
        if "admin_veto" in record:
            verdict = "admin:reject"
        if decision is not None:
            route = decision["route"]
            if decision.get("pocket") is not None:
                route += ":" + decision["pocket"]
        text = " ".join(record["text"].split())
        change = record.get("change", {"kind": "-", "targets": []})
        targets = []
        for target in change["targets"]:
            targets.append(target["id"])
        linked = ",".join(targets) or "-"
        lines.append(f"{record['id']:16}  {record['status']:8}  {record['novelty']:10}  {verdict:14}  {route:7}  "
                     f"{record['source']['kind']:12}  {change['kind']:8}  {linked:16}  {text}")
    return "\n".join(lines)
