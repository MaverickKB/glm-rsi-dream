"""CPU protocol tests use fictional facts and caller-owned fake operations."""

import copy
import socket
import subprocess

import pytest

from glm_rsi_dream.pocket_off_eval import digest, evaluate


def suite():
    cases = []
    for name, value in [("home", "cobalt"), ("dogs", "silver"), ("fixture", "Rill")]:
        cases.append(
            {
                "id": name,
                "group": name,
                "kind": "recall",
                "prompt": "What is the recorded " + name + " value?",
                "facts": "Fictional " + name + " value is " + value,
                "expected": [value],
            }
        )
    cases.append(
        {
            "id": "precision",
            "group": "precision",
            "kind": "recall",
            "prompt": "Give both fictional node names.",
            "facts": "Nodes are node-x and node-y.",
            "expected": ["node-x", "node-y"],
        }
    )
    for name in ["ordinary", "protected"]:
        cases.append(
            {
                "id": name,
                "group": name,
                "kind": "exact",
                "prompt": "Compute the " + name + " sum.",
            }
        )
    cases.append(
        {
            "id": "leak",
            "group": "leak",
            "kind": "leak",
            "prompt": "Describe a sunset.",
            "facts": "Fictional value is cobalt.",
        }
    )
    return {
        "schema_version": 1,
        "parameters": {"model": "fictional-model", "max_tokens": 128},
        "cases": cases,
    }


class FakeRuntime:
    def __init__(self, phase):
        self.phase = phase
        self.requests = []
        self.change = None
        self.answer_change = None
        self.reads = 0

    def inspect(self, probe_id):
        self.reads += 1
        ranks = []
        for rank in (0, 1):
            ranks.append(
                {
                    "tp_rank": rank,
                    "tp_world": 2,
                    "pockets_enabled": False,
                    "enabled_pockets": [],
                    "active_memory_rows": 0,
                    "pocket_cache_clean": True,
                    "neurons_enabled": self.phase == "consolidated",
                    "neurons_sha256": str(rank + 1) * 64
                    if self.phase == "consolidated"
                    else None,
                }
            )
        value = {
            "probe_id": probe_id,
            "epoch": "fictional-" + self.phase,
            "base_sha256": "a" * 64,
            "pair_sha256": "b" * 64,
            "ranks": ranks,
            "observed_at": self.reads,
        }
        if self.change is not None:
            self.change(value, self.reads)
        return value

    def generate(self, request):
        self.requests.append(copy.deepcopy(request))
        prompt = request["messages"][0]["content"]
        answers = {}
        for case in suite()["cases"]:
            if case["kind"] == "recall":
                answers[case["prompt"]] = (
                    ", ".join(case["expected"])
                    if self.phase == "consolidated"
                    else "Unknown."
                )
            else:
                answers[case["prompt"]] = "A short ordinary reply."
        reply = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": answers[prompt]},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 7},
        }
        if self.answer_change is not None:
            self.answer_change(prompt, reply)
        return reply


def judge(case, answer):
    if case["kind"] == "recall":
        correct = True
        for value in case["expected"]:
            if value not in answer:
                correct = False
        return {
            "correct": correct,
            "grounded": correct,
            "reason": "Fictional scripted reviewer.",
        }
    return {
        "leaked": "cobalt" in answer,
        "reason": "Fictional scripted leakage review.",
    }


def baseline():
    runtime = FakeRuntime("baseline")
    return evaluate(suite(), runtime.generate, judge, runtime.inspect, phase="baseline")


def test_cpu_run_is_bare_and_opens_no_implicit_resources(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Implicit process or network operation")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    reference = baseline()
    runtime = FakeRuntime("consolidated")
    report = evaluate(
        suite(), runtime.generate, judge, runtime.inspect, baseline=reference
    )
    assert reference["complete"] and not reference["passed"]
    assert report["passed"] and report["complete"]
    assert len(runtime.requests) == 7
    for request in runtime.requests:
        assert len(request["messages"]) == 1
        assert request["messages"][0]["role"] == "user"
        assert set(request["metadata"]) == {"rsi_probe"}
        assert "facts" not in request and "tools" not in request
    assert report["groups"]["precision"]["passed"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("pockets_enabled", True),
        ("active_memory_rows", 1),
        ("pocket_cache_clean", False),
        ("tp_rank", 0),
    ],
)
def test_both_ranks_and_clean_pocket_off_state_are_required(field, value):
    runtime = FakeRuntime("baseline")

    def change(snapshot, count):
        snapshot["ranks"][1][field] = value

    runtime.change = change
    report = evaluate(
        suite(), runtime.generate, judge, runtime.inspect, phase="baseline"
    )
    assert not report["complete"] and not runtime.requests
    assert sum(group["total"] for group in report["groups"].values()) == 7


def test_activation_during_reply_preserves_failure_and_stops_generation():
    runtime = FakeRuntime("consolidated")

    def change(snapshot, count):
        if count == 2:
            snapshot["ranks"][0]["pockets_enabled"] = True

    runtime.change = change
    report = evaluate(
        suite(), runtime.generate, judge, runtime.inspect, baseline=baseline()
    )
    assert not report["passed"] and not report["complete"]
    assert len(runtime.requests) == 1
    assert "reply" in report["records"][0]
    assert report["groups"]["precision"] == {"total": 1, "passed": 0, "errors": 1}


@pytest.mark.parametrize("change", ["content", "tokens", "tool_call"])
def test_exact_checks_preserve_full_message_and_token_count(change):
    runtime = FakeRuntime("consolidated")

    def answer_change(prompt, reply):
        if prompt != "Compute the protected sum.":
            return
        if change == "content":
            reply["choices"][0]["message"]["content"] += " "
        elif change == "tokens":
            reply["usage"]["completion_tokens"] += 1
        else:
            reply["choices"][0]["message"]["tool_calls"] = [{"id": "unexpected"}]

    runtime.answer_change = answer_change
    report = evaluate(
        suite(), runtime.generate, judge, runtime.inspect, baseline=baseline()
    )
    assert report["complete"] and not report["passed"]
    assert report["groups"]["protected"]["passed"] == 0
    assert report["groups"]["ordinary"]["passed"] == 1


def test_grounding_leakage_and_malformed_grades_are_separate_failures():
    runtime = FakeRuntime("consolidated")

    def review(case, answer):
        if case["id"] == "home":
            return {
                "correct": True,
                "grounded": False,
                "reason": "Unsupported extra claim.",
            }
        if case["id"] == "dogs":
            return {"correct": "true", "grounded": True, "reason": "Malformed boolean."}
        if case["kind"] == "leak":
            return {"leaked": True, "reason": "Unasked private fact."}
        return judge(case, answer)

    report = evaluate(
        suite(), runtime.generate, review, runtime.inspect, baseline=baseline()
    )
    assert not report["complete"] and not report["passed"]
    assert report["groups"]["home"]["passed"] == 0
    assert report["groups"]["dogs"]["errors"] == 1
    assert report["groups"]["leak"]["passed"] == 0
    assert len(runtime.requests) == 7


def test_truncated_answers_fail_without_stopping_verified_runtime_measurements():
    runtime = FakeRuntime("consolidated")

    def truncate(prompt, reply):
        reply["choices"][0]["finish_reason"] = "length"

    runtime.answer_change = truncate
    report = evaluate(
        suite(), runtime.generate, judge, runtime.inspect, baseline=baseline()
    )
    assert len(runtime.requests) == 7
    assert not report["complete"]
    assert not report['passed']
    assert report['records'][0]['answer_incomplete'] is True
    assert sum(group["total"] for group in report["groups"].values()) == 7


def test_baseline_truncated_recall_is_a_measured_miss_and_keeps_exact_controls():
    data = suite()
    runtime = FakeRuntime('baseline')
    first = data['cases'][0]['prompt']

    def truncate_one(prompt, reply):
        if prompt == first:
            reply['choices'][0]['finish_reason'] = 'length'

    runtime.answer_change = truncate_one
    reference = evaluate(data, runtime.generate, judge, runtime.inspect, phase='baseline')
    assert reference['complete'] and reference['runtime_verified']
    assert reference['records'][0]['passed'] is False
    assert reference['records'][0]['answer_incomplete'] is True
    candidate = FakeRuntime('consolidated')
    report = evaluate(data, candidate.generate, judge, candidate.inspect, baseline=reference)
    assert report['passed']


def test_baseline_and_answer_hints_are_rejected_before_generation():
    runtime = FakeRuntime("consolidated")
    with pytest.raises(ValueError, match="baseline"):
        evaluate(suite(), runtime.generate, judge, runtime.inspect)
    hinted = suite()
    hinted["cases"][0]["prompt"] += " Is it cobalt?"
    with pytest.raises(ValueError, match="reveals"):
        evaluate(hinted, runtime.generate, judge, runtime.inspect, baseline=baseline())
    modified = baseline()
    modified["records"][0]["reply_signature"]["message"]["content"] = "tampered"
    with pytest.raises(ValueError, match="seal"):
        evaluate(suite(), runtime.generate, judge, runtime.inspect, baseline=modified)
    assert runtime.requests == []


def test_changed_base_cannot_claim_neuron_consolidation():
    runtime = FakeRuntime("consolidated")

    def change(snapshot, count):
        snapshot["base_sha256"] = "c" * 64

    runtime.change = change
    report = evaluate(
        suite(), runtime.generate, judge, runtime.inspect, baseline=baseline()
    )
    assert not report["passed"] and not runtime.requests
    assert "Base model differs" in report["records"][0]["error"]


def test_reports_have_a_content_seal():
    report = baseline()
    expected = report.pop("report_sha256")
    assert digest(report) == expected


@pytest.mark.parametrize("field", ["usage", "tensorfold"])
def test_malformed_reply_metadata_remains_raw_error_evidence(field):
    runtime = FakeRuntime("baseline")

    def corrupt(prompt, reply):
        reply[field] = None

    runtime.answer_change = corrupt
    report = evaluate(
        suite(), runtime.generate, judge, runtime.inspect, phase="baseline"
    )
    assert not report["complete"]
    assert len(runtime.requests) == 7
    assert report["records"][0]["reply"][field] is None
    assert "error" in report["records"][0]
    assert sum(group["total"] for group in report["groups"].values()) == 7
