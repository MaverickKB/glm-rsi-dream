"""Report GLM agreement with a sealed labeled sanity set on the Spark head."""

import argparse
import hashlib
import json
import socket
from pathlib import Path
from typing import Any

from .grade import grade_pending, validate_policy
from .queue import append_record, candidate_id, latest_records
from .timing import now_et, summarize_timings


def route_metrics(assessments: Any) -> Any:
    """Compare labels with GLM outputs; ungraded items remain false negatives."""
    confusion: dict[Any, Any] = {}
    pairs = []
    for row in assessments:
        expected = row["expected"]
        actual_label = "reject" if expected["verdict"] == "reject" else expected["route"]
        grade = row["grade"]
        predicted_label = "ungraded"
        if grade is not None:
            predicted_label = "reject" if grade["verdict"] == "reject" else grade["route"]
        counts = confusion.setdefault(actual_label, {})
        counts[predicted_label] = counts.get(predicted_label, 0) + 1
        pairs.append((actual_label, predicted_label))
    metrics = {}
    for label in ("reject", "weights", "pocket", "memory", "learned"):
        true_positive = 0
        false_positive = 0
        false_negative = 0
        for actual, predicted in pairs:
            actual_matches = actual == label
            predicted_matches = predicted == label
            if label == "learned":
                actual_matches = actual in ("weights", "pocket")
                predicted_matches = predicted in ("weights", "pocket")
            if actual_matches and predicted_matches:
                true_positive += 1
            elif predicted_matches:
                false_positive += 1
            elif actual_matches:
                false_negative += 1
        support = true_positive + false_negative
        predicted_count = true_positive + false_positive
        precision = true_positive / predicted_count if predicted_count else None
        recall = true_positive / support if support else None
        metrics[label] = {
            "true_positive": true_positive, "false_positive": false_positive,
            "false_negative": false_negative, "support": support,
            "predicted": predicted_count, "precision": precision, "recall": recall,
        }
    return {"per_route": metrics, "confusion": confusion}


def run_sanity(cases_path: Any, root: Any, *, endpoint: Any, model: Any, timeout: Any = 120, policy: Any = None) -> Any:
    cases_path = Path(cases_path)
    root = Path(root)
    raw = cases_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    cases = json.loads(raw)
    if not isinstance(cases, list) or not cases:
        raise ValueError("evaluation cases must be a nonempty array")
    settings = validate_policy(policy)
    root.mkdir(parents=True, exist_ok=True)
    seal_path = root / "sanity-seal.json"
    report_path = root / "agreement.json"
    seal = {
        "cases_sha256": digest,
        "grade_source_sha256": hashlib.sha256(Path(__file__).with_name("grade.py").read_bytes()).hexdigest(),
        "runner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "grade_policy": settings,
    }
    if seal_path.exists():
        if json.loads(seal_path.read_text()) != seal:
            raise ValueError("sanity data or grader changed after sealing; preserve this run")
    else:
        seal_path.write_text(json.dumps(seal, indent=2) + "\n")
    if report_path.exists():
        raise ValueError("sanity run already scored; use a new experiment root")
    queue_path = root / "queue" / "candidates.jsonl"
    seen = set()
    for record in latest_records(queue_path):
        seen.add(record["id"])
    labels = {}
    for index, case in enumerate(cases, start=1):
        expected = case["expected"]
        if expected["verdict"] not in ("accept", "reject"):
            raise ValueError("expected verdict must be accept or reject")
        if expected["verdict"] == "accept" and expected["route"] not in ("weights", "pocket", "memory"):
            raise ValueError("expected accepted route must be weights, pocket or memory")
        # Case titles can disclose expected classes; only an opaque number reaches GLM.
        source_ref = "grader-sanity:case-" + str(index)
        identifier = candidate_id(source_ref, case["text"])
        if identifier in labels:
            raise ValueError("duplicate sanity case")
        labels[identifier] = case
        if identifier in seen:
            continue
        record = {
            "id": identifier, "created_et": now_et(),
            "source": {"kind": "prompt", "ref": source_ref},
            "text": case["text"], "prompt_evidence": case["prompt_evidence"],
            "glm_prior": case["glm_prior"], "novelty": case["novelty"],
            "status": "pending", "grade": None,
            "timings_ms": {"extract": 0, "probe": 0, "grade": 0, "absorb": 0},
        }
        append_record(queue_path, record)
    result = grade_pending(root, endpoint=endpoint, model=model, timeout=timeout, policy=policy)
    verdict_matches = 0
    route_matches = 0
    route_total = 0
    joint_matches = 0
    assessments = []
    for record in latest_records(queue_path):
        case = labels[record["id"]]
        expected = case["expected"]
        grade = record["grade"]
        verdict_match = grade is not None and grade["verdict"] == expected["verdict"]
        route_match = None
        if expected["verdict"] == "accept":
            route_total += 1
            route_match = (
                verdict_match and grade["route"] == expected["route"]
                and grade["pocket"] == expected["pocket"]
            )
            route_matches += int(route_match)
        joint_match = verdict_match and route_match is not False
        verdict_matches += int(verdict_match)
        joint_matches += int(joint_match)
        assessments.append({
            "name": case["name"], "id": record["id"], "expected": expected,
            "grade": grade, "wall_ms": record["timings_ms"]["grade"],
            "verdict_match": verdict_match, "route_match": route_match,
            "joint_match": joint_match,
        })
    report = {
        **seal, "observed_et": now_et(), "host": socket.gethostname(),
        "model": model, "total": len(cases),
        "verdict_agreement": {"matches": verdict_matches, "total": len(cases)},
        "accepted_route_agreement": {"matches": route_matches, "total": route_total},
        "joint_agreement": {"matches": joint_matches, "total": len(cases)},
        "timings": summarize_timings(root / "timings.jsonl"),
        "errors": result["errors"], "assessments": assessments,
        "dataset_role": "development",
        "scope": "Labeled grading hypotheses with supplied priors; reusable across new configured experiments. Not a live novelty-probe or generalization benchmark.",
    }
    report.update(route_metrics(assessments))
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main(argv: Any = None) -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8888")
    parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args(argv)
    report = run_sanity(args.cases, args.root, endpoint=args.endpoint, model=args.model, timeout=args.timeout)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
