"""Positive and adversarial examples for the offline M2/M3 acceptance checks."""

import copy

import pytest

from tools.dream_eval_checks import check_attribution, check_candidate, digest


def report(phase, correct):
    value = {"phase": phase, "complete": True, "runtime_verified": True,
             "suite_sha256": "a" * 64, "base_sha256": "b" * 64,
             "records": [{"id": "target", "kind": "recall", "judgment": {
                 "correct": correct, "grounded": True, "reason": "Fixture review."}}]}
    value["report_sha256"] = digest(value)
    return value


def test_m2_real_score_gain_with_off_ablation_passes_the_score_check():
    result = check_attribution(report("baseline", False), report("consolidated", True),
                               report("baseline", False), ["target"])
    assert result == {"recovered_and_ablated": ["target"], "native_acceptance": False}


@pytest.mark.parametrize("before,after,off", [(True, True, True), (False, True, True),
                                           (False, False, False), ("false", True, False)])
def test_m2_no_gain_no_ablation_or_malformed_grade_cannot_establish_learning(before, after, off):
    with pytest.raises(ValueError):
        check_attribution(report("baseline", before), report("consolidated", after),
                          report("baseline", off), ["target"])


def test_m2_report_tampering_is_rejected():
    consolidated = report("consolidated", True)
    consolidated["base_sha256"] = "c" * 64
    with pytest.raises(ValueError):
        check_attribution(report("baseline", False), consolidated, report("baseline", False), ["target"])


def lock():
    return {"passing_control_candidate": True, "selected": {
        "correct_grounded": 1, "total": 24, "exact_controls": 2, "total_controls": 2,
        "controls": [{"id": "ordinary", "exact": True}, {"id": "protected", "exact": True}]}}


def test_m3_gain_and_complete_controls_open_development_eligibility():
    assert check_candidate(lock(), 0, 1) == {"development_eligible": True, "native_acceptance": False}


@pytest.mark.parametrize("mutation", ["failed-control-flag", "no-gain", "lost-control",
                                     "missing-control", "string-control-flag", "duplicate-control"])
def test_m3_diagnostic_failures_cannot_be_promoted(mutation):
    value = copy.deepcopy(lock())
    if mutation == "failed-control-flag":
        value["passing_control_candidate"] = False
    elif mutation == "no-gain":
        value["selected"]["correct_grounded"] = 0
    elif mutation == "lost-control":
        value["selected"]["exact_controls"] = 1
    elif mutation == "missing-control":
        value["selected"]["controls"].pop()
    elif mutation == "string-control-flag":
        value["passing_control_candidate"] = "true"
    else:
        value["selected"]["controls"][1]["id"] = "ordinary"
    with pytest.raises(ValueError):
        check_candidate(value, 0, 1)
