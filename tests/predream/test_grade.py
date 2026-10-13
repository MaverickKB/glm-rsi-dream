"""Tripwires for protocol handling and append-only grading, without local judgments."""

import copy
import hashlib
import json

import pytest

from glm_rsi_dream.predream.grade import (
    GradeError,
    grade_candidate,
    grade_pending,
    grade_prompt,
    parse_grade,
    validate_policy,
)


def decision(**changes):
    value = {
        "verdict": "accept", "route": "pocket", "pocket": "workshop",
        "reason": "Source states a reusable workshop setting.", "score": 0.9,
    }
    value.update(changes)
    return value


def candidate(source_ref="fictional-sanity"):
    text = "The fictional workshop inspection interval is 41 days."
    identifier = hashlib.sha256((source_ref + text).encode()).hexdigest()[:16]
    return {
        "id": identifier, "created_et": "2026-10-12T21:45:00-04:00",
        "source": {"kind": "prompt", "ref": source_ref},
        "text": "The fictional workshop inspection interval is 41 days.",
        "prompt_evidence": "The fictional workshop inspection interval is 41 days.",
        "glm_prior": {"answer": "I do not know this workshop's interval.", "agrees": False},
        "novelty": "new", "status": "pending", "grade": None,
        "timings_ms": {"extract": 1.0, "probe": 2.0, "grade": 0.0, "absorb": 0.0},
    }


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "Decision:\n{}\nFinished."])
def test_single_json_object_and_wrappers(wrapper):
    raw = wrapper.format(json.dumps(decision()))
    assert parse_grade(raw) == decision(grader="glm")


@pytest.mark.parametrize("changes", [
    {"verdict": "maybe"}, {"route": "disk"}, {"pocket": None},
    {"pocket": "../escape"}, {"pocket": "Upper"}, {"pocket": "a" * 65},
    {"route": "memory", "pocket": "workshop"}, {"reason": " "},
    {"score": True}, {"score": "0.9"}, {"score": -0.1}, {"score": 1.1},
    {"score": float("nan")}, {"score": float("inf")}, {"grader": "local"},
    {"score": 10 ** 500},
])
def test_invalid_protocol_is_rejected(changes):
    with pytest.raises(ValueError):
        parse_grade(json.dumps(decision(**changes)))


def test_duplicate_fields_ambiguous_objects_and_malformed_json():
    valid = json.dumps(decision())
    duplicate = valid.replace('"score": 0.9', '"score": 0.9, "score": 0.1')
    for raw in (duplicate, valid + "\n" + valid, "{broken " + valid, "", "null"):
        with pytest.raises(ValueError):
            parse_grade(raw)


def test_repair_retry_requests_same_data_without_local_fallback(monkeypatch):
    monkeypatch.setattr("glm_rsi_dream.predream.grade.time.sleep", lambda duration: None)
    requests = []

    def client(prompt):
        requests.append(prompt)
        if len(requests) == 1:
            return '{"verdict":"accept"}'
        return json.dumps(decision())

    record = candidate()
    original = copy.deepcopy(record)
    grade = grade_candidate(record, client)
    assert grade == decision(grader="glm")
    assert record == original
    assert record["text"] in requests[0] and record["text"] in requests[1]
    assert "failed the output protocol" in requests[1]


def test_transport_failure_has_bounded_attempts(monkeypatch):
    monkeypatch.setattr("glm_rsi_dream.predream.grade.time.sleep", lambda duration: None)
    requests = []

    def offline(prompt):
        requests.append(prompt)
        raise OSError("head unavailable")

    with pytest.raises(GradeError, match="after 3 attempts"):
        grade_candidate(candidate(), offline)
    assert len(requests) == 3


def test_only_pending_and_bounded_retry_configuration():
    record = candidate()
    record["status"] = "accepted"
    with pytest.raises(ValueError, match="pending"):
        grade_candidate(record, None)
    for attempts in (0, 4, True, 1.5):
        with pytest.raises(ValueError, match="max_attempts"):
            grade_candidate(candidate(), None, max_attempts=attempts)


def test_queue_appends_decisions_and_does_not_regrade_terminal_rows(tmp_path):
    from glm_rsi_dream.predream.queue import append_record, latest_records

    path = tmp_path / "queue" / "candidates.jsonl"
    record = candidate()
    append_record(path, record)
    rejected = candidate("fictional-sanity-second")
    append_record(path, rejected)
    requests = []

    def client(prompt):
        requests.append(prompt)
        if len(requests) == 1:
            return json.dumps(decision())
        return json.dumps(decision(verdict="reject", route="weights", pocket=None))

    result = grade_pending(tmp_path, client)
    rows = latest_records(path)
    assert len(result["graded"]) == 2 and not result["errors"]
    assert [row["status"] for row in rows] == ["accepted", "rejected"]
    assert all(row["grade"]["grader"] == "glm" for row in rows)
    assert all(row["timings_ms"]["grade"] >= 0 for row in rows)
    assert len(path.read_text().splitlines()) == 4
    assert len((tmp_path / "timings.jsonl").read_text().splitlines()) == 2
    assert grade_pending(tmp_path, client) == {"graded": [], "errors": []}
    assert len(requests) == 2


def test_failed_grade_preserves_pending_and_stops_batch(tmp_path, monkeypatch):
    from glm_rsi_dream.predream.queue import append_record, latest_records

    monkeypatch.setattr("glm_rsi_dream.predream.grade.time.sleep", lambda duration: None)
    path = tmp_path / "queue" / "candidates.jsonl"
    record = candidate()
    append_record(path, record)
    second = candidate("fictional-sanity-second")
    append_record(path, second)
    requests = []

    def client(prompt):
        requests.append(prompt)
        return "invalid"

    result = grade_pending(tmp_path, client)
    assert not result["graded"] and len(result["errors"]) == 1
    assert len(requests) == 3
    assert latest_records(path) == [record, second]
    timing = json.loads((tmp_path / "timings.jsonl").read_text())
    assert timing["stage"] == "grade" and "error" in timing


def test_sanity_prompts_hide_labels_and_case_titles(tmp_path, monkeypatch):
    from glm_rsi_dream.predream.grade_sanity import run_sanity

    record = candidate()
    cases = [{
        "name": "EXPECTED_REJECT_UNSUPPORTED_SECRET_LABEL",
        "text": record["text"], "prompt_evidence": record["prompt_evidence"],
        "glm_prior": record["glm_prior"], "novelty": record["novelty"],
        "expected": {"verdict": "reject", "route": None, "pocket": None},
    }]
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(cases))
    requests = []

    def chat(self, messages):
        requests.append(messages[0]["content"])
        return json.dumps(decision(verdict="reject"))

    monkeypatch.setattr("glm_rsi_dream.predream.extract.GLMClient.chat", chat)
    report = run_sanity(cases_path, tmp_path / "run", endpoint="unused", model="stub")
    assert report["joint_agreement"] == {"matches": 1, "total": 1}
    assert len(requests) == 1
    assert "EXPECTED_REJECT_UNSUPPORTED_SECRET_LABEL" not in requests[0]
    assert '"expected"' not in requests[0]
    assert "grader-sanity:case-1" not in requests[0]
    assert '"source"' not in requests[0]
    with pytest.raises(ValueError, match="already scored"):
        run_sanity(cases_path, tmp_path / "run", endpoint="unused", model="stub")


def test_rejected_pocket_needs_no_target_and_is_not_regraded(tmp_path):
    from glm_rsi_dream.predream.queue import append_record, latest_records

    value = decision(verdict="reject", pocket=None, reason="The value is absent from the evidence.")
    assert parse_grade(json.dumps(value)) == {**value, "grader": "glm"}
    record = candidate()
    path = tmp_path / "queue" / "candidates.jsonl"
    append_record(path, record)
    result = grade_pending(tmp_path, lambda prompt: json.dumps(value))
    assert not result["errors"]
    latest = latest_records(path)[0]
    assert latest["status"] == "rejected" and latest["grade"]["pocket"] is None
    assert grade_pending(tmp_path, None) == {"graded": [], "errors": []}


def test_custom_policy_swaps_rubric_but_preserves_protocol_and_data():
    policy = {"strategy": "custom", "instructions": "Hypothesis: keep all supported claims in memory."}
    prompt = grade_prompt(candidate(), policy)
    assert "Hypothesis: keep all supported claims in memory." in prompt
    assert "Apply these checks IN ORDER" not in prompt
    assert "Return exactly one JSON object" in prompt
    assert candidate()["text"] in prompt
    routing = grade_prompt(candidate(), {"routing_rules": "All retained items use memory."})
    assert "All retained items use memory." in routing
    assert "4. Destination: durable general reusable knowledge" not in routing


@pytest.mark.parametrize("policy", [
    {"strategy": "unknown"}, {"strategy": "custom"}, {"instructions": "custom without strategy"},
    {"routing_rules": ""}, {"max_attempts": 4}, {"max_attempts": True}, {"typo": "value"},
    {"strategy": "custom", "instructions": "A rubric", "routing_rules": "Another rubric"},
])
def test_policy_typos_and_incomplete_strategies_fail_before_generation(policy):
    with pytest.raises(ValueError):
        validate_policy(policy)


def test_configured_retry_limit_is_applied_without_inventing_a_grade(monkeypatch):
    requests = []

    def client(prompt):
        requests.append(prompt)
        return "invalid"

    with pytest.raises(GradeError, match="after 1 attempts"):
        grade_candidate(candidate(), client, policy={"max_attempts": 1})
    assert len(requests) == 1


def test_per_route_metrics_include_false_positives_and_ungraded_items():
    from glm_rsi_dream.predream.grade_sanity import route_metrics

    rows = [
        {"expected": {"verdict": "accept", "route": "weights"}, "grade": decision(route="memory", pocket=None)},
        {"expected": {"verdict": "accept", "route": "memory"}, "grade": decision(route="memory", pocket=None)},
        {"expected": {"verdict": "accept", "route": "pocket"}, "grade": None},
        {"expected": {"verdict": "reject", "route": None}, "grade": decision(verdict="reject", pocket=None)},
    ]
    result = route_metrics(rows)
    assert result["per_route"]["memory"]["precision"] == 0.5
    assert result["per_route"]["memory"]["recall"] == 1
    assert result["per_route"]["weights"]["precision"] is None
    assert result["per_route"]["weights"]["recall"] == 0
    assert result["per_route"]["pocket"]["false_negative"] == 1
    assert result["per_route"]["reject"]["precision"] == 1
    assert result["confusion"]["pocket"]["ungraded"] == 1
