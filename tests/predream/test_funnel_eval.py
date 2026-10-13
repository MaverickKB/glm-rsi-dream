import json

import pytest

from glm_rsi_dream.predream.funnel_eval import match_facts, run_evaluation, score
from glm_rsi_dream.predream.queue import append_record, candidate_id, latest_records


class Client:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def chat(self, messages, **options):
        self.requests.append(messages)
        return next(self.replies)


def case():
    return {"source": "The zither tally is the sum plus nine.",
            "expected": [{"id": "e1", "text": "The zither tally is the sum plus nine.",
                          "route": "weights", "novelty": "new"}]}


def grade():
    return {"verdict": "accept", "route": "weights", "pocket": None,
            "reason": "Supported new reusable definition.", "score": 0.9}


def test_actual_funnel_and_handoff_with_labels_withheld(tmp_path):
    item = case()
    text = item["source"]
    extracted = {"candidates": [{"text": text, "prompt_evidence": text,
                                 "probe_question": "How is the zither tally defined?"}]}
    client = Client([json.dumps(extracted), "I do not know.",
                     json.dumps({"agrees": None, "novelty": "new", "reason": "Unknown prior."}),
                     json.dumps(grade())])
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([item]))
    root = tmp_path / "run"
    report = run_evaluation(path, root, client=client)
    assert report["per_route"]["weights"]["precision"] == 1
    assert report["per_route"]["weights"]["recall"] == 1
    assert report["missing"] == report["extras"] == 0
    assert report["replay_unchanged"] is True
    assert latest_records(root / "queue/candidates.jsonl")[0]["status"] == "staged"
    assert len(client.requests) == 4
    for messages in client.requests[:4]:
        assert '"expected_facts"' not in json.dumps(messages)
    assert text not in json.dumps(client.requests[1])
    assert report["timings"]["extract"]["count"] == 1
    assert report["timings"]["probe"]["count"] == 1
    assert report["timings"]["grade"]["count"] == 1
    with pytest.raises(ValueError, match="fresh"):
        run_evaluation(path, root, client=client)


def test_missing_merged_and_extra_acceptances_count_as_errors():
    first = case()
    second = {"source": "Water is H2O.", "expected": [{"id": "e2", "text": "Water is H2O.",
                                                      "route": "reject", "novelty": "known"}]}
    record = {"id": "extra", "grade": grade(), "novelty": "new"}
    result = score([first, second], [record], [{"matches": []}])
    assert result["missing"] == 2 and result["extras"] == 1
    assert result["false_acceptances"] == 1
    assert result["per_route"]["weights"]["fn"] == 1
    assert result["per_route"]["weights"]["fp"] == 1
    assert result["per_route"]["reject"]["fn"] == 1
    assert result["confusion"] == {"weights": {"missing": 1}, "reject": {"missing": 1},
                                   "unexpected": {"weights": 1}}


def test_partial_claims_cannot_match_whole_gold_fact_and_labels_are_withheld():
    item = {"source": "Ada approved collection and withheld publication.",
            "expected": [{"id": "approval", "text": "Ada approved collection and withheld publication.",
                          "route": "memory", "novelty": "new"}]}
    candidates = [{"id": "c1", "text": "Ada approved collection."},
                  {"id": "c2", "text": "Ada withheld publication."}]
    client = Client([json.dumps({"same_fact": False, "reason": "Only half the full fact."})] * 2)
    result = match_facts(item, candidates, client)
    assert result == {"matches": []}
    assert len(client.requests) == 2
    for messages in client.requests:
        payload = json.loads(messages[1]["content"])
        assert set(payload) == {"candidate", "gold"}
        assert payload["gold"] == item["expected"][0]["text"]


def test_exact_matches_and_duplicate_candidates_keep_one_to_one_without_model_calls():
    item = case()
    candidates = [{"id": "c1", "text": item["source"]},
                  {"id": "c2", "text": item["source"]}]
    client = Client([])
    result = match_facts(item, candidates, client)
    assert result["matches"] == [{"expected_id": "e1", "candidate_id": "c1",
                                   "reason": "Exact text equality."}]
    assert client.requests == []


@pytest.mark.parametrize("reference", ["old:1", "case-1"])
def test_live_queue_transfer_preserves_parent_and_original_provenance(tmp_path, reference):
    item = case()
    text = item["source"]
    identifier = candidate_id(reference, text)
    record = {"id": identifier, "created_et": "2026-10-12T22:30:00-04:00",
              "source": {"kind": "prompt", "ref": reference}, "text": text,
              "prompt_evidence": text, "glm_prior": {"answer": "Unknown", "agrees": None},
              "novelty": "new", "status": "pending", "grade": None,
              "timings_ms": {"extract": 1, "probe": 2, "grade": 0, "absorb": 0}}
    parent = tmp_path / "parent.jsonl"
    append_record(parent, record)
    before = parent.read_bytes()
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([item]))
    client = Client([json.dumps(grade())])
    root = tmp_path / "transfer"
    report = run_evaluation(path, root, client=client, queue_from=parent)
    assert report["per_route"]["weights"]["recall"] == 1
    assert parent.read_bytes() == before
    copied = latest_records(root / "queue/candidates.jsonl")[0]
    assert copied["id"] == identifier and copied["source"]["ref"] == reference
    assert copied["glm_prior"] == record["glm_prior"]
    assert "extract" not in report["timings"] and "probe" not in report["timings"]


def test_declared_source_ref_preserves_missing_targets_when_transfer_is_empty(tmp_path):
    item = case()
    item["source_ref"] = "intake:missing-document-version"
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps([item]))
    parent = tmp_path / "empty.jsonl"
    parent.write_text("")
    report = run_evaluation(cases_path, tmp_path / "empty-transfer", client=Client([]), queue_from=parent)
    assert report["missing"] == 1
    assert report["per_route"]["weights"]["fn"] == 1
    assert report["confusion"] == {"weights": {"missing": 1}}


def test_transfer_matches_declared_source_identity_instead_of_ordinal(tmp_path):
    item = case()
    reference = "case-22"
    item["source_ref"] = reference
    text = item["source"]
    identifier = candidate_id(reference, text)
    record = {"id": identifier, "created_et": "2026-10-13T01:00:00-04:00",
              "source": {"kind": "prompt", "ref": reference}, "text": text,
              "prompt_evidence": text, "glm_prior": {"answer": "Unknown", "agrees": None},
              "novelty": "new", "status": "pending", "grade": None,
              "timings_ms": {"extract": 1, "probe": 2, "grade": 0, "absorb": 0}}
    parent = tmp_path / "parent.jsonl"
    append_record(parent, record)
    original = parent.read_bytes()
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps([item]))
    report = run_evaluation(cases_path, tmp_path / "declared", client=Client([json.dumps(grade())]), queue_from=parent)
    assert report["missing"] == report["extras"] == 0
    assert report["per_route"]["weights"]["recall"] == 1
    assert parent.read_bytes() == original
