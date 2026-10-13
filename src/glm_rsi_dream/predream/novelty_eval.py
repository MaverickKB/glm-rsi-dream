"""Evaluate configurable novelty policies against labels withheld from GLM."""

import hashlib
import json
from pathlib import Path
from typing import Any

from ..files import write_json
from .metrics import class_metrics
from .novelty import judge_novelty, resolve_policy
from .timing import StageTimer, now_et, summarize_timings


def run_evaluation(cases_path: Any, root: Any, *, client: Any, policy: Any = None) -> Any:
    cases_path = Path(cases_path)
    root = Path(root)
    raw = cases_path.read_bytes()
    cases = json.loads(raw)
    if not isinstance(cases, list) or not cases:
        raise ValueError("Novelty evaluation needs a nonempty labeled list")
    for case in cases:
        for field in ("text", "probe_question"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError("Novelty case needs " + field)
        if case.get("expected") not in ("known", "new", "conflicts"):
            raise ValueError("Novelty label must be known, new or conflicts")
    settings = resolve_policy(policy)
    seal = {"cases_sha256": hashlib.sha256(raw).hexdigest(), "policy": settings,
            "novelty_source_sha256": hashlib.sha256(Path(__file__).with_name("novelty.py").read_bytes()).hexdigest()}
    seal_path = root / "novelty-eval-seal.json"
    if seal_path.exists() and json.loads(seal_path.read_text()) != seal:
        raise ValueError("Novelty evaluation changed; use a new run root")
    if (root / "novelty-evaluation.json").exists():
        raise ValueError("Evaluation already completed; preserve it and use a new root")
    write_json(seal_path, seal)
    expected = []
    predicted: list[Any] = []
    assessments = []
    errors: list[Any] = []
    for number, case in enumerate(cases, start=1):
        identifier = "case-" + str(number)
        if errors:
            expected.append(case["expected"])
            predicted.append(None)
            assessments.append({"id": identifier, "expected": case["expected"], "predicted": None,
                                "judgment": None, "wall_ms": None, "error": "Not attempted after stage failure"})
            continue
        judgment = None
        error_text = None
        try:
            with StageTimer(root, "probe", identifier) as timer:
                judgment = judge_novelty({"text": case["text"], "probe_question": case["probe_question"]}, client, settings)
        except (ValueError, RuntimeError, OSError) as error:
            error_text = str(error)
            errors.append({"id": identifier, "error": error_text})
        prediction = None
        if judgment is not None:
            prediction = judgment["comparison"]["novelty"]
        expected.append(case["expected"])
        predicted.append(prediction)
        assessments.append({"id": identifier, "expected": case["expected"], "predicted": prediction,
                            "judgment": judgment, "wall_ms": timer.wall_ms, "error": error_text})
        write_json(root / "novelty-eval-progress.json", {"assessments": assessments, "errors": errors})
    report = {"observed_et": now_et(), "seal": seal, "total": len(cases),
              "per_class": class_metrics(expected, predicted), "assessments": assessments,
              "timings": summarize_timings(root / "timings.jsonl"), "errors": errors,
              "scope": "Labeled development evaluation; repeating this set measures policy changes, not fresh generalization."}
    write_json(root / "novelty-evaluation.json", report)
    return report
