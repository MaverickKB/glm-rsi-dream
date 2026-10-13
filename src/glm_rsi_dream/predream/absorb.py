"""Persist approved personal memory and seal raw facts for a manual dream."""

import fcntl
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..files import digest, read_json, sha, write_json
from .control import authorize_handoff
from .queue import append_record, latest_records
from .review import verify_targets
from .timing import StageTimer, now_et, record_timing


@contextmanager
def absorption_lock(root: Any) -> Any:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / ".absorb.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def approved_route(record: Any) -> Any:
    grade = record.get("grade")
    if not isinstance(grade, dict):
        raise ValueError("Accepted item has no GLM grade: " + record["id"])
    if grade.get("grader") != "glm" or grade.get("verdict") != "accept":
        raise ValueError("Accepted item requires a GLM accept verdict: " + record["id"])
    if record["source"]["kind"] == "manual":
        review = record.get("review", {})
        if review.get("safe") is not True or review.get("reviewer") != "glm":
            raise ValueError("Manual item requires a safe GLM review: " + record["id"])
    route = grade.get("route")
    if route not in ("weights", "pocket", "memory"):
        raise ValueError("Unknown accepted route: " + str(route))
    pocket = grade.get("pocket")
    if route == "pocket":
        if not isinstance(pocket, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", pocket):
            raise ValueError("Pocket name must use lowercase letters, digits, hyphens or underscores")
    elif pocket is not None:
        raise ValueError("Only pocket-routed items may name a pocket")
    return route


def content_identity(record: Any) -> Any:
    content = {}
    for key in ("id", "created_et", "source", "text", "prompt_evidence", "glm_prior", "novelty", "grade"):
        content[key] = record[key]
    for key in ("intake", "review", "change"):
        if key in record:
            content[key] = record[key]
    return digest(content)


def read_memory(path: Any) -> Any:
    records: dict[Any, Any] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            item_id = record["id"]
            if item_id in records:
                raise ValueError("Duplicate personal-memory id: " + item_id)
            records[item_id] = record
    return records


def persist_memory(path: Any, record: Any) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def absorb(root: Any) -> Any:
    """Write memory before its queue receipt, so interrupted calls can resume."""
    root = Path(root)
    queue_path = root / "queue" / "candidates.jsonl"
    memory_path = root / "memory" / "memory.jsonl"
    start_et = now_et()
    started = time.perf_counter()
    absorbed = []
    recovered = []
    with absorption_lock(root):
        memory = read_memory(memory_path)
        for record in latest_records(queue_path):
            if record["status"] != "accepted":
                continue
            if approved_route(record) != "memory":
                continue
            authorize_handoff(root, record)
            verify_targets(root, record)
            item_id = record["id"]
            identity = content_identity(record)
            existing = memory.get(item_id)
            if existing is not None and content_identity(existing) != identity:
                raise ValueError("Personal-memory content changed for id: " + item_id)
            with StageTimer(root, "absorb", item_id) as timer:
                updated = dict(record)
                updated["status"] = "absorbed"
                if existing is None:
                    stored = dict(updated)
                    stored["absorbed_et"] = now_et()
                    persist_memory(memory_path, stored)
                    memory[item_id] = stored
                    absorbed.append(item_id)
                else:
                    recovered.append(item_id)
                updated["timings_ms"] = dict(record["timings_ms"])
                updated["timings_ms"]["absorb"] = (time.perf_counter() - timer.started) * 1000
                append_record(queue_path, updated)
    if not absorbed and not recovered:
        record_timing(root, "absorb", "run", start_et, (time.perf_counter() - started) * 1000)
    return {"absorbed": absorbed, "recovered": recovered, "memory_path": str(memory_path)}


def stage_directory(root: Any, run_id: Any) -> Any:
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ValueError("run_id must be a single safe directory name")
    root = Path(root).resolve()
    directory = root / "staged" / run_id
    if not directory.resolve().is_relative_to(root):
        raise ValueError("Staging directory escapes the run root")
    return directory


def verify_manifest(root: Any, directory: Any, manifest: Any) -> Any:
    if manifest.get("schema_version") != 1 or manifest.get("run_id") != directory.name:
        raise ValueError("Staged manifest identity mismatch")
    sealed = dict(manifest)
    expected = sealed.pop("manifest_sha256", None)
    if digest(sealed) != expected:
        raise ValueError("Staged manifest hash mismatch")
    current = {}
    for record in latest_records(Path(root) / "queue" / "candidates.jsonl"):
        current[record["id"]] = record
    for item in manifest["items"]:
        record = current.get(item["id"])
        if record is None or record["status"] not in ("accepted", "staged"):
            raise ValueError("Staged item is no longer approved: " + item["id"])
        approved_route(record)
        authorize_handoff(root, record)
        verify_targets(root, record)
        if content_identity(record) != item["content_sha256"]:
            raise ValueError("Staged candidate changed: " + item["id"])
    for relative, expected in manifest["files_sha256"].items():
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError("Missing or unsafe staged input: " + relative)
        if sha(path) != expected:
            raise ValueError("Staged input hash mismatch: " + relative)


def mark_staged(queue_path: Any, records: Any) -> Any:
    for record in records:
        if record["status"] == "staged":
            continue
        updated = dict(record)
        updated["status"] = "staged"
        append_record(queue_path, updated)


def stage(root: Any, run_id: Any) -> Any:
    """Seal per-route raw statements and facts; publish before queue receipts."""
    root = Path(root)
    directory = stage_directory(root, run_id)
    queue_path = root / "queue" / "candidates.jsonl"
    with absorption_lock(root), StageTimer(root, "stage", run_id):
        records = latest_records(queue_path)
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            verify_manifest(root, directory, manifest)
            staged_ids = set()
            for item in manifest["items"]:
                staged_ids.add(item["id"])
            resume = []
            for record in records:
                if record["status"] != "accepted":
                    continue
                if approved_route(record) == "memory":
                    continue
                if record["id"] not in staged_ids:
                    raise ValueError("Staging run is sealed; use a new run_id for new accepted facts")
                resume.append(record)
            mark_staged(queue_path, resume)
            return manifest

        selected = []
        groups: dict[Any, Any] = {}
        accepted_ids: dict[Any, Any] = {"weights": [], "pocket": {}}
        items = []
        for record in records:
            if record["status"] != "accepted":
                continue
            route = approved_route(record)
            if route == "memory":
                continue
            authorize_handoff(root, record)
            verify_targets(root, record)
            pocket = record["grade"]["pocket"]
            group_key = (route, pocket)
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(record)
            selected.append(record)
            if route == "weights":
                accepted_ids["weights"].append(record["id"])
            else:
                if pocket not in accepted_ids["pocket"]:
                    accepted_ids["pocket"][pocket] = []
                accepted_ids["pocket"][pocket].append(record["id"])
            item = {"id": record["id"], "text": record["text"], "source": record["source"],
                    "grade": record["grade"], "content_sha256": content_identity(record)}
            if "change" in record:
                item["change"] = record["change"]
            items.append(item)

        inputs = []
        files_sha256 = {}
        for number, (group_key, group_records) in enumerate(groups.items(), start=1):
            route, pocket = group_key
            statements = []
            facts = []
            ids = []
            for record in group_records:
                statements.append(record["text"])
                facts.append({"id": record["id"], "statement": record["text"]})
                ids.append(record["id"])
            statement_file = f"inputs/{number:04d}-statement.json"
            facts_file = f"inputs/{number:04d}-facts.json"
            write_json(directory / statement_file, {"statement": "\n".join(statements)})
            write_json(directory / facts_file, {"facts": facts})
            files_sha256[statement_file] = sha(directory / statement_file)
            files_sha256[facts_file] = sha(directory / facts_file)
            inputs.append({"route": route, "pocket": pocket, "ids": ids,
                           "statement_file": statement_file, "facts_file": facts_file})
        manifest = {
            "schema_version": 1, "run_id": run_id, "created_et": now_et(),
            "readiness": "raw_facts", "accepted_ids": accepted_ids,
            "items": items, "inputs": inputs, "files_sha256": files_sha256,
            "consumer_formats": {
                "statement": "ATTNMEM-pocket RSI input.json: {statement: text}",
                "facts": "glm_rsi_dream.dataset lesson facts: [{id, statement}]",
            },
            "required_preparation": [
                "GLM authors and reviews fresh training/development/heldout splits and scope routes",
                "Freeze lesson/evaluation and protected controls before numerical capture",
                "Capture and review train-only trajectories before fitting on both ranks",
            ],
        }
        manifest["manifest_sha256"] = digest(manifest)
        write_json(manifest_path, manifest)
        mark_staged(queue_path, selected)
        return manifest


def dream_plan(root: Any, run_id: Any = None) -> Any:
    """Read verified staged inputs without importing or invoking a trainer."""
    root = Path(root)
    directories = []
    if run_id is not None:
        directories.append(stage_directory(root, run_id))
    else:
        for path in sorted((root / "staged").glob("*/manifest.json")):
            if not path.parent.name.startswith("."):
                directories.append(stage_directory(root, path.parent.name))
    runs = []
    for directory in directories:
        manifest = read_json(directory / "manifest.json")
        verify_manifest(root, directory, manifest)
        groups = []
        for entry in manifest["inputs"]:
            groups.append({"route": entry["route"], "pocket": entry["pocket"], "ids": entry["ids"],
                           "statement": str(directory / entry["statement_file"]),
                           "facts": str(directory / entry["facts_file"])})
        runs.append({"run_id": manifest["run_id"], "manifest": str(directory / "manifest.json"),
                     "readiness": manifest["readiness"], "groups": groups,
                     "required_preparation": manifest["required_preparation"]})
    return {"mode": "manual", "training_executed": False, "human_confirm_required": True,
            "runs": runs,
            "next_action": "Human reviews raw inputs and authorizes prepared training with --confirm"}
