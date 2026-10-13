"""Evaluate source collection and a preserved live queue through the real funnel."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ..files import write_json
from .absorb import absorb, stage
from .extract import extract_candidates, write_jsonl
from .grade import grade_pending
from .metrics import class_metrics
from .queue import append_record, format_table, latest_records
from .timing import StageTimer, now_et, summarize_timings

ROUTES = ("reject", "weights", "pocket", "memory")



def load_cases(path: Any) -> Any:
    cases = json.loads(Path(path).read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("Funnel cases require a nonempty list")
    identifiers = set()
    source_refs = set()
    for number, case in enumerate(cases, 1):
        if not isinstance(case.get("source"), str) or not case["source"].strip():
            raise ValueError("Funnel case requires source text")
        if not isinstance(case.get("expected"), list):
            raise ValueError("Funnel case requires expected facts")
        reference = case.get("source_ref", "case-" + str(number))
        if not isinstance(reference, str) or not reference.strip() or reference in source_refs:
            raise ValueError("Funnel source references must be unique nonempty strings")
        source_refs.add(reference)
        for fact in case["expected"]:
            if not isinstance(fact.get("id"), str) or not fact["id"] or fact["id"] in identifiers:
                raise ValueError("Expected fact ids must be unique nonempty strings")
            if not isinstance(fact.get("text"), str) or not fact["text"].strip():
                raise ValueError("Expected fact requires text")
            if fact.get("route") not in ROUTES or fact.get("novelty") not in ("known", "new", "conflicts"):
                raise ValueError("Expected fact requires route and novelty labels")
            identifiers.add(fact["id"])
    return cases


def match_facts(case: Any, candidates: Any, client: Any) -> Any:
    """Use shared full-fact scoring with deterministic one-to-one bookkeeping."""
    from .extraction_eval import align_candidates

    gold = []
    for fact in case["expected"]:
        gold.append({"name": fact["id"], "text": fact["text"]})
    alignment = align_candidates(candidates, gold, client=client)
    matches = []
    for match in alignment["matches"]:
        matches.append({"expected_id": match["gold_id"],
                        "candidate_id": match["candidate_id"], "reason": match["reason"]})
    return {"matches": matches}


def source_number(reference: Any) -> Any:
    """Read ordinals from current evaluator refs and preserved fixture refs."""
    if reference.startswith("case-"):
        value = reference[5:]
    else:
        value = reference.rsplit(":", 1)[-1]
    number = int(value)
    if number < 1:
        raise ValueError("Evaluation source ordinal must be positive")
    return number


def case_index(reference: Any, cases: Any) -> Any:
    for index, case in enumerate(cases):
        if reference == case.get("source_ref", "case-" + str(index + 1)):
            return index
    for case in cases:
        if "source_ref" in case:
            raise ValueError("Parent queue source is absent from the declared cases")
    index = source_number(reference) - 1
    if index >= len(cases):
        raise ValueError("Parent queue source is absent from the supplied cases")
    return index


def score(cases: Any, records: Any, alignments: Any, probe_records: Any = None) -> Any:
    expected_routes = []
    predicted_routes = []
    expected_novelty = []
    predicted_novelty = []
    confusion: dict[Any, Any] = {}
    assessments = []
    missing = 0
    extras = 0
    false_acceptances = 0
    by_id = {}
    used = set()
    matches = {}
    priors = {}
    for record in probe_records or records:
        priors[record["id"]] = record["novelty"]
    for record in records:
        by_id[record["id"]] = record
    for alignment in alignments:
        for match in alignment["matches"]:
            matches[match["expected_id"]] = match["candidate_id"]
    for case in cases:
        for fact in case["expected"]:
            record = by_id.get(matches.get(fact["id"]))
            prediction = "missing"
            novelty = None
            if record is None:
                missing += 1
            else:
                used.add(record["id"])
                novelty = priors.get(record["id"])
                grade = record["grade"]
                prediction = "ungraded"
                if grade is not None:
                    prediction = "reject" if grade["verdict"] == "reject" else grade["route"]
            if fact["route"] == "reject" and prediction in ("weights", "pocket", "memory"):
                false_acceptances += 1
            expected_routes.append(fact["route"])
            predicted_routes.append(prediction)
            expected_novelty.append(fact["novelty"])
            predicted_novelty.append(novelty)
            assessments.append({"expected_id": fact["id"], "candidate_id": None if record is None else record["id"],
                                "expected": fact["route"], "predicted": prediction})
    for record in records:
        if record["id"] in used:
            continue
        extras += 1
        grade = record["grade"]
        prediction = "ungraded"
        if grade is not None:
            prediction = "reject" if grade["verdict"] == "reject" else grade["route"]
            if grade["verdict"] == "accept":
                false_acceptances += 1
        expected_routes.append("unexpected")
        predicted_routes.append(prediction)
        assessments.append({"expected_id": None, "candidate_id": record["id"],
                            "expected": "unexpected", "predicted": prediction})
    for expected, predicted in zip(expected_routes, predicted_routes, strict=True):
        row = confusion.setdefault(expected, {})
        row[predicted] = row.get(predicted, 0) + 1
    metrics = class_metrics(expected_routes, predicted_routes)
    per_route = {}
    f1_total = 0
    for route in ROUTES:
        value = metrics.get(route, {"tp": 0, "fp": 0, "fn": 0, "precision": None, "recall": None})
        denominator = 2 * value["tp"] + value["fp"] + value["fn"]
        value["f1"] = 2 * value["tp"] / denominator if denominator else None
        per_route[route] = value
        f1_total += value["f1"] or 0
    learned_expected = []
    learned_predicted = []
    for expected, predicted in zip(expected_routes, predicted_routes, strict=True):
        learned_expected.append("learned" if expected in ("weights", "pocket") else expected)
        learned_predicted.append("learned" if predicted in ("weights", "pocket") else predicted)
    learned = class_metrics(learned_expected, learned_predicted).get("learned")
    return {"per_route": per_route, "learned": learned, "macro_f1": f1_total / len(ROUTES),
            "confusion": confusion, "missing": missing, "extras": extras,
            "false_acceptances": false_acceptances, "assessments": assessments,
            "novelty_per_class": class_metrics(expected_novelty, predicted_novelty)}


def run_evaluation(cases_path: Any, root: Any, *, client: Any, policy: Any = None, queue_from: Any = None, learned_facts: Any = None) -> Any:
    root = Path(root)
    if (root / "funnel-evaluation.json").exists() or (root / "queue/candidates.jsonl").exists():
        raise ValueError("Use a fresh funnel root; preserve the completed or partial experiment")
    cases = load_cases(cases_path)
    settings = policy or {"extract": {}, "grade": {}}
    seal = {"cases_sha256": hashlib.sha256(Path(cases_path).read_bytes()).hexdigest(), "policy": settings}
    queue_path = root / "queue/candidates.jsonl"
    started = time.perf_counter()
    errors = []
    if learned_facts is not None:
        from .manual import import_learned

        import_learned(root, learned_facts)
        seal["catalog_sha256"] = hashlib.sha256(Path(learned_facts).read_bytes()).hexdigest()
    if queue_from is not None:
        raw = Path(queue_from).read_bytes()
        seal["parent_queue_sha256"] = hashlib.sha256(raw).hexdigest()
        seen = set()
        highest = 0
        for line in raw.decode().splitlines():
            record = json.loads(line)
            if record["id"] in seen:
                continue
            if record["status"] == "learned":
                continue
            if record["status"] != "pending" or record["grade"] is not None:
                raise ValueError("Parent queue must preserve its original pending snapshots")
            index = case_index(record["source"]["ref"], cases)
            highest = max(highest, index + 1)
            append_record(queue_path, record)
            seen.add(record["id"])
        explicit_sources = False
        for case in cases:
            if "source_ref" in case:
                explicit_sources = True
        if not explicit_sources:
            cases = cases[:highest]
    write_json(root / "funnel-seal.json", seal)
    if queue_from is None:
        for number, case in enumerate(cases, start=1):
            try:
                reference = case.get("source_ref", "case-" + str(number))
                extract_candidates(case["source"], reference, client=client,
                                   source_kind=case.get("source_kind", "prompt"), queue_path=queue_path,
                                   timings_path=root / "timings.jsonl", policy=settings["extract"])
            except (ValueError, RuntimeError, OSError) as error:
                errors.append({"stage": "extract", "case": number, "error": str(error)})
                break
    probe_records = []
    for record in latest_records(queue_path):
        if record["status"] != "learned":
            probe_records.append(record)
    write_json(root / "collected-candidates.json", probe_records)
    result = grade_pending(root, client, policy=settings["grade"])
    errors.extend(result["errors"])
    records = []
    for record in latest_records(queue_path):
        if record["status"] != "learned":
            records.append(record)
    write_json(root / "funnel-candidates.json", records)
    for status in (None, "accepted", "rejected", "pending"):
        with StageTimer(root, "queue_list", status or "all"):
            selected = []
            for record in latest_records(queue_path):
                if status is None or record["status"] == status:
                    selected.append(record)
            (root / ("queue-" + (status or "all") + ".txt")).write_text(format_table(selected) + "\n")
    funnel_ms = (time.perf_counter() - started) * 1000
    alignments = []
    for number, case in enumerate(cases, start=1):
        selected = []
        for record in records:
            source_ref = record["source"]["ref"]
            same_case = source_ref == case.get("source_ref", "case-" + str(number))
            if queue_from is not None:
                same_case = case_index(source_ref, cases) == number - 1
            if same_case:
                selected.append(record)
        try:
            with StageTimer(root, "alignment", "case-" + str(number)):
                alignment = match_facts(case, selected, client)
            alignments.append(alignment)
            write_jsonl(root / "alignment.jsonl", {"case": number, **alignment})
        except (ValueError, RuntimeError, OSError) as error:
            errors.append({"stage": "alignment", "case": number, "error": str(error)})
            break
    report = {"observed_et": now_et(), "seal": seal, "errors": errors, "candidates": len(records),
              "funnel_ms": funnel_ms, "scope": "Development full-funnel evaluation with GLM post-collection semantic alignment."}
    report["awaiting_admin"] = result.get("awaiting_approval", [])
    report["acceptance_complete"] = not errors and not report["awaiting_admin"]
    report.update(score(cases, records, alignments, probe_records))
    report["handoff"] = {"memory": absorb(root), "staged": stage(root, root.name)}
    preserved = {}
    paths = (queue_path, root / "memory/memory.jsonl", root / "staged" / root.name / "manifest.json")
    for path in paths:
        if path.exists():
            preserved[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    if report["acceptance_complete"]:
        replay = grade_pending(root, client, policy=settings["grade"])
        absorb(root)
        stage(root, root.name)
        for path, digest in preserved.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("Replay changed a completed queue or handoff artifact")
        if replay["graded"] or replay["errors"]:
            raise ValueError("Completed queue replay produced additional grades")
        report["replay_unchanged"] = True
    report["timings"] = summarize_timings(root / "timings.jsonl")
    write_json(root / "funnel-evaluation.json", report)
    return report
