"""Offline M2/M3 score checks. A passing check is not native acceptance."""

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def recall_scores(report, targets):
    if report.get("complete") is not True or report.get("runtime_verified") is not True:
        raise ValueError("Causal comparison requires complete runtime-verified reports")
    sealed = copy.deepcopy(report)
    declared = sealed.pop("report_sha256", None)
    if digest(sealed) != declared:
        raise ValueError("Report seal changed")
    scores = {}
    seen = set()
    for row in report["records"]:
        if row["id"] in seen:
            raise ValueError("Duplicate report case")
        seen.add(row["id"])
        if row["id"] not in targets:
            continue
        if row.get("kind") != "recall" or row.get("error"):
            raise ValueError("Declared target lacks a valid recall observation")
        grade = row["judgment"]
        if type(grade.get("correct")) is not bool or type(grade.get("grounded")) is not bool:
            raise ValueError("Recall flags must be Boolean")
        if not isinstance(grade.get("reason"), str) or not grade["reason"].strip():
            raise ValueError("Recall grade lacks a reason")
        scores[row["id"]] = grade["correct"] and grade["grounded"]
    if set(scores) != set(targets):
        raise ValueError("Missing declared target")
    return scores


def check_attribution(baseline, consolidated, ablation, targets):
    if not isinstance(targets, list) or not targets:
        raise ValueError("Targets must be a nonempty unique predeclared ID list")
    seen = set()
    for identifier in targets:
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("Target ID must be a nonempty string")
        seen.add(identifier)
    for field in ("suite_sha256", "base_sha256"):
        identity = baseline.get(field)
        if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity):
            raise ValueError("Missing baseline identity: " + field)
        if consolidated.get(field) != identity or ablation.get(field) != identity:
            raise ValueError("Causal arms use different " + field)
    if baseline.get("phase") != "baseline" or consolidated.get("phase") != "consolidated" or ablation.get("phase") != "baseline":
        raise ValueError("Expected OFF baseline, ON consolidated and OFF ablation arms")
    before = recall_scores(baseline, targets)
    after = recall_scores(consolidated, targets)
    removed = recall_scores(ablation, targets)
    recovered = []
    for identifier in targets:
        if before[identifier]:
            continue
        if not after[identifier]:
            raise ValueError("Declared baseline miss was not recovered: " + identifier)
        if removed[identifier]:
            raise ValueError("Recovered target persists with neurons OFF: " + identifier)
        recovered.append(identifier)
    if not recovered:
        raise ValueError("Zero baseline-missed targets, so no learning gain is established")
    return {"recovered_and_ablated": recovered, "native_acceptance": False}


def check_candidate(lock, baseline_recall, minimum_recall):
    if type(baseline_recall) is not int or baseline_recall < 0:
        raise ValueError("Baseline recall must be a nonnegative integer")
    if type(minimum_recall) is not int or minimum_recall <= 0:
        raise ValueError("A positive predeclared recall minimum is required")
    if lock.get("passing_control_candidate") is not True:
        raise ValueError("Diagnostic candidate has no passing development controls")
    selected = lock["selected"]
    for field in ("correct_grounded", "total", "exact_controls", "total_controls"):
        if type(selected.get(field)) is not int:
            raise ValueError("Candidate counter must be an integer: " + field)
    total = selected["total"]
    recall = selected["correct_grounded"]
    if total <= 0 or not 0 <= recall <= total or minimum_recall > total:
        raise ValueError("Candidate recall counters or minimum are invalid")
    if recall < minimum_recall or recall <= baseline_recall:
        raise ValueError("Candidate lacks the declared development gain")
    controls = selected.get("controls", [])
    if selected["total_controls"] <= 0 or selected["exact_controls"] != selected["total_controls"]:
        raise ValueError("Candidate lost development controls")
    if len(controls) != selected["total_controls"]:
        raise ValueError("Candidate control observations are incomplete")
    identifiers = set()
    for row in controls:
        if row.get("exact") is not True or not isinstance(row.get("id"), str) or row["id"] in identifiers:
            raise ValueError("Candidate control row is invalid or duplicated")
        identifiers.add(row["id"])
    return {"development_eligible": True, "native_acceptance": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    attribution = commands.add_parser("attribution")
    for name in ("baseline", "consolidated", "ablation", "targets"):
        attribution.add_argument("--" + name, type=Path, required=True)
    candidate = commands.add_parser("candidate")
    candidate.add_argument("--selected", type=Path, required=True)
    candidate.add_argument("--baseline-recall", type=int, required=True)
    candidate.add_argument("--minimum-recall", type=int, required=True)
    args = parser.parse_args()
    try:
        if args.command == "attribution":
            result = check_attribution(json.loads(args.baseline.read_text()), json.loads(args.consolidated.read_text()),
                                       json.loads(args.ablation.read_text()), json.loads(args.targets.read_text()))
        else:
            result = check_candidate(json.loads(args.selected.read_text()), args.baseline_recall, args.minimum_recall)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(json.dumps({"passed": False, "native_acceptance": False, "reason": str(error)}))
        raise SystemExit(1) from None
    print(json.dumps({"passed": True, **result}))


if __name__ == "__main__":
    main()
