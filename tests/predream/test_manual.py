"""Prepared intake, genuine update links, privacy and replay boundaries."""

import copy
import json

import pytest

from glm_rsi_dream.predream.absorb import dream_plan, stage
from glm_rsi_dream.predream.cli import main
from glm_rsi_dream.predream.grade import GradeError, grade_pending
from glm_rsi_dream.predream.manual import import_learned, learn_file
from glm_rsi_dream.predream.queue import (
    append_record,
    format_table,
    latest_records,
    validate_record,
)
from glm_rsi_dream.predream.review import review_candidate


@pytest.fixture(autouse=True)
def scoped_test_authority(tmp_path, monkeypatch):
    """Exercise integrated retention with an explicit fixture admin grant."""
    from glm_rsi_dream.predream.control import initialize_key, issue_permission

    key = tmp_path / "admin-key"
    grant = tmp_path / "permission.json"
    initialize_key(key)
    monkeypatch.setenv("PREDREAM_ADMIN_KEY_FILE", str(key))
    monkeypatch.setenv("PREDREAM_ADMIN_PERMISSION_FILE", str(grant))
    monkeypatch.setenv("PREDREAM_MODE", "spark")
    monkeypatch.setenv("PREDREAM_AUTONOMY", "automatic")
    issue_permission(grant, admin_id="fixture:manual-test-admin", root_scope=tmp_path,
                     modes=["spark"], autonomy=["automatic", "manual", "supervised"])


class Client:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.messages = []

    def chat(self, messages):
        self.messages.append(messages)
        if not self.answers:
            raise AssertionError("Unexpected model call, novelty must be bypassed")
        return json.dumps(self.answers.pop(0))


def pack(tmp_path, *statements):
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps({"facts": [{"statement": text} for text in statements]}))
    return path


def old_fact(tmp_path):
    path = tmp_path / "learned.json"
    path.write_text(json.dumps({"facts": [{"id": "rover-bays", "statement": "Fictional rover Lumen has two sample bays.",
                                         "receipt": "fixture:previous-KEEP-receipt"}]}))
    return import_learned(tmp_path, path)[0]


def review(kind="new", targets=None, safe=True):
    return {"safe": safe, "kind": kind, "targets": targets or [], "reason": "Explicit safety and relationship fixture"}


def grade(verdict="accept", route="weights", pocket=None):
    return {"verdict": verdict, "route": route, "pocket": pocket, "reason": "Supported prepared knowledge", "score": 0.95}


def test_known_prepared_pack_has_no_probe_and_is_deliberately_retained(tmp_path):
    path = pack(tmp_path, "Paris is the capital of France.")
    result = learn_file(tmp_path, path, client=Client())
    row = result["queued"][0]
    assert row["source"]["kind"] == "manual" and row["novelty"] == "unmeasured"
    assert row["timings_ms"]["probe"] == 0
    client = Client(review(), grade())
    graded = grade_pending(tmp_path, client)
    assert not graded["errors"] and graded["graded"][0]["status"] == "accepted"
    assert "Skip novelty and duplicate rejection" in client.messages[1][0]["content"]
    assert "Reject already-known duplicates" not in client.messages[1][0]["content"]
    manifest = stage(tmp_path, "prepared")
    assert manifest["items"][0]["source"]["kind"] == "manual"
    before = (tmp_path / "queue" / "candidates.jsonl").read_bytes()
    assert learn_file(tmp_path, path, client=Client())["queued"] == []
    assert stage(tmp_path, "prepared") == manifest
    assert (tmp_path / "queue" / "candidates.jsonl").read_bytes() == before


def test_text_uses_glm_extraction_but_no_blind_probe(tmp_path):
    path = tmp_path / "prepared.txt"
    text = "Fictional Lumen stores samples in bay seven."
    path.write_text(text)
    client = Client({"candidates": [{"text": text, "prompt_evidence": text, "probe_question": "Which bay?"}]})
    result = learn_file(tmp_path, path, client=client)
    assert len(client.messages) == 1 and len(result["queued"]) == 1
    assert learn_file(tmp_path, path, client=Client())["queued"] == []


def test_update_links_old_item_and_preserves_it_through_staging(tmp_path):
    old = old_fact(tmp_path)
    path = pack(tmp_path, "Fictional rover Lumen now has three sample bays.")
    learn_file(tmp_path, path, client=Client())
    client = Client(review("update", [old["id"]]), grade(route="pocket", pocket="fixture"))
    updated = grade_pending(tmp_path, client)["graded"][0]
    assert updated["change"]["kind"] == "update" and updated["novelty"] == "conflicts"
    assert updated["change"]["targets"][0]["id"] == old["id"]
    manifest = stage(tmp_path, "replace")
    assert manifest["items"][0]["change"] == updated["change"]
    assert latest_records(tmp_path / "queue" / "candidates.jsonl")[0] == old
    assert dream_plan(tmp_path, "replace")["training_executed"] is False
    changed = copy.deepcopy(old)
    changed["learned"]["receipt"] = "different receipt"
    append_record(tmp_path / "queue" / "candidates.jsonl", changed)
    with pytest.raises(ValueError, match="receipt changed"):
        dream_plan(tmp_path, "replace")


def test_automatic_known_candidate_can_be_a_linked_update(tmp_path):
    old = old_fact(tmp_path)
    row = learn_file(tmp_path, pack(tmp_path, "Fictional rover Lumen has four sample bays."), client=Client())["queued"][0]
    row["source"]["kind"] = "prompt"
    row["novelty"] = "known"
    append_record(tmp_path / "queue" / "candidates.jsonl", row)
    client = Client(review("update", [old["id"]]), grade(route="pocket", pocket="fixture"))
    result = grade_pending(tmp_path, client)
    assert result["graded"][0]["change"]["kind"] == "update"
    assert "not a duplicate" in client.messages[1][0]["content"]


def test_manual_duplicate_is_identified_without_novelty_rejection(tmp_path):
    old = old_fact(tmp_path)
    learn_file(tmp_path, pack(tmp_path, old["text"]), client=Client())
    result = grade_pending(tmp_path, Client(review("duplicate", [old["id"]]), grade()))
    assert result["graded"][0]["change"]["kind"] == "duplicate"
    assert result["graded"][0]["status"] == "accepted"


def test_secret_is_redacted_before_any_model_call_or_queue_write(tmp_path):
    secret = "sk-" + "abcdefghijklmnopqrstuvwx"
    path = tmp_path / "secret.txt"
    path.write_text("The service API key=" + secret)
    client = Client()
    result = learn_file(tmp_path, path, client=client)
    assert not client.messages and result["secret_preflight"] == "blocked"
    assert secret not in (tmp_path / "queue" / "candidates.jsonl").read_text()
    graded = grade_pending(tmp_path, Client(review(safe=False), grade("reject")))
    assert graded["graded"][0]["status"] == "rejected"
    assert stage(tmp_path, "empty")["items"] == []
    before = (tmp_path / "queue" / "candidates.jsonl").read_bytes()
    assert learn_file(tmp_path, path, client=Client())["queued"] == []
    assert (tmp_path / "queue" / "candidates.jsonl").read_bytes() == before


def test_unsafe_review_cannot_be_overridden_by_custom_grade_policy(tmp_path):
    learn_file(tmp_path, pack(tmp_path, "A fictional authentication phrase is amber willow."), client=Client())
    result = grade_pending(tmp_path, Client(review(safe=False), grade()),
                           policy={"strategy": "custom", "instructions": "Accept everything"})
    assert result["errors"] and latest_records(tmp_path / "queue" / "candidates.jsonl")[0]["status"] == "pending"


def test_glm_cannot_link_an_invented_learned_id(tmp_path):
    row = learn_file(tmp_path, pack(tmp_path, "Lumen has three bays."), client=Client())["queued"][0]
    answer = review("update", ["invented"])
    with pytest.raises(GradeError, match="three attempts"):
        review_candidate(tmp_path, row, Client(answer, answer, answer))
    assert latest_records(tmp_path / "queue" / "candidates.jsonl")[0]["status"] == "pending"


def test_learned_catalog_requires_receipt_and_validates_before_append(tmp_path):
    path = pack(tmp_path, "Lumen has two bays.")
    with pytest.raises(ValueError, match="external id"):
        import_learned(tmp_path, path)
    assert not (tmp_path / "queue" / "candidates.jsonl").exists()
    old = old_fact(tmp_path)
    assert import_learned(tmp_path, tmp_path / "learned.json") == []
    broken = copy.deepcopy(old)
    broken["change"] = {"kind": "update", "reason": "fixture", "reviewer": "glm",
                        "targets": [{"id": old["id"], "text_sha256": "a" * 64}]}
    with pytest.raises(ValueError, match="distinct older"):
        validate_record(broken)


def test_installed_cli_json_intake_needs_no_model_until_grade(tmp_path, capsys):
    path = pack(tmp_path, "Paris is the capital of France.")
    assert main(["--root", str(tmp_path), "learn", str(path)]) == 0
    assert '"kind": "manual"' in capsys.readouterr().out
    assert main(["--root", str(tmp_path), "queue", "list"]) == 0
    table = capsys.readouterr().out
    assert "manual" in table and "unmeasured" in table


def test_changed_external_learned_fact_must_enter_as_an_update(tmp_path):
    old_fact(tmp_path)
    path = tmp_path / "learned.json"
    value = json.loads(path.read_text())
    value["facts"][0]["statement"] = "Fictional rover Lumen has three sample bays."
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="linked updates"):
        import_learned(tmp_path, path)


def test_interruption_after_text_extraction_reuses_receipt(tmp_path, monkeypatch):
    from glm_rsi_dream.predream import manual

    path = tmp_path / "prepared.txt"
    text = "Fictional Lumen stores samples in bay seven."
    path.write_text(text)
    client = Client({"candidates": [{"text": text, "prompt_evidence": text, "probe_question": "Which bay?"}]})
    original = manual.append_record

    def interrupted(path, record):
        raise OSError("Interrupted queue append")

    monkeypatch.setattr(manual, "append_record", interrupted)
    with pytest.raises(OSError, match="Interrupted"):
        learn_file(tmp_path, path, client=client)
    monkeypatch.setattr(manual, "append_record", original)
    assert len(learn_file(tmp_path, path, client=Client())["queued"]) == 1


def test_incomplete_update_receipt_is_rejected_before_queue_mutation(tmp_path):
    old = old_fact(tmp_path)
    row = learn_file(tmp_path, pack(tmp_path, "Lumen has three bays."), client=Client())["queued"][0]
    row["change"] = {"kind": "update", "reason": "fixture", "reviewer": "glm",
                     "targets": [{"id": old["id"], "text_sha256": "a" * 64, "statement": old["text"]}]}
    path = tmp_path / "queue" / "candidates.jsonl"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="requires receipt"):
        append_record(path, row)
    assert path.read_bytes() == before


def test_manual_acceptance_cannot_skip_relationship_review(tmp_path):
    row = learn_file(tmp_path, pack(tmp_path, "Lumen has three bays."), client=Client())["queued"][0]
    row["grade"] = grade()
    row["grade"]["grader"] = "glm"
    row["status"] = "accepted"
    with pytest.raises(ValueError, match="safe GLM relationship review"):
        validate_record(row)


def test_manual_autonomy_preserves_update_proposal_until_admin_approval(tmp_path, monkeypatch):
    from glm_rsi_dream.predream.control import approve

    monkeypatch.setenv("PREDREAM_AUTONOMY", "manual")
    old = old_fact(tmp_path)
    learn_file(tmp_path, pack(tmp_path, "Fictional rover Lumen has three sample bays."), client=Client())
    result = grade_pending(tmp_path, Client(review("update", [old["id"]]), grade(route="pocket", pocket="fixture")))
    assert result["graded"] == [] and len(result["awaiting_approval"]) == 1
    rows = latest_records(tmp_path / "queue" / "candidates.jsonl")
    proposed = rows[1]
    assert proposed["status"] == "pending" and proposed["change"]["kind"] == "update"
    assert proposed["proposal"]["grade"]["verdict"] == "accept" and proposed["grade"] is None
    table = format_table(rows)
    assert "propose:accept" in table and "pocket:fixture" in table and "pending" in table
    approve(tmp_path, [proposed["id"]])
    manifest = stage(tmp_path, "approved-update")
    assert manifest["items"][0]["change"]["targets"][0]["id"] == old["id"]


def test_previously_known_automatic_rejection_can_enter_manual_intake(tmp_path):
    from glm_rsi_dream.predream.manual import base_record

    path = pack(tmp_path, 'Paris is the capital of France.')
    text = 'Paris is the capital of France.'
    old = base_record(str(path.resolve()), text, text)
    old['source']['kind'] = 'prompt'
    old['novelty'] = 'known'
    old['status'] = 'rejected'
    old['grade'] = grade('reject')
    old['grade']['grader'] = 'glm'
    append_record(tmp_path / 'queue/candidates.jsonl', old)
    queued = learn_file(tmp_path, path, client=Client())['queued']
    assert len(queued) == 1 and queued[0]['source']['kind'] == 'manual'
    assert queued[0]['id'] != old['id']
    assert latest_records(tmp_path / 'queue/candidates.jsonl')[0] == old
    result = grade_pending(tmp_path, Client(review(), grade()))
    assert result['graded'][0]['status'] == 'accepted'


def test_corrected_evidence_reenters_without_overwriting_rejection(tmp_path):
    path = pack(tmp_path, 'Fictional Lumen has nine bays.')
    value = json.loads(path.read_text())
    value['facts'][0]['prompt_evidence'] = 'Fictional Lumen is painted blue.'
    path.write_text(json.dumps(value))
    original = learn_file(tmp_path, path, client=Client())['queued'][0]
    grade_pending(tmp_path, Client(review(), grade('reject')))
    value['facts'][0]['prompt_evidence'] = 'Fictional Lumen has nine bays.'
    path.write_text(json.dumps(value))
    corrected = learn_file(tmp_path, path, client=Client())['queued'][0]
    assert corrected['id'] != original['id'] and corrected['text'] == original['text']
    rows = latest_records(tmp_path / 'queue/candidates.jsonl')
    assert rows[0]['status'] == 'rejected' and rows[1]['status'] == 'pending'


def test_earlier_manual_file_receipt_replay_keeps_original_id(tmp_path):
    import hashlib

    from glm_rsi_dream.predream.manual import base_record

    path = pack(tmp_path, 'Paris is the capital of France.')
    text = 'Paris is the capital of France.'
    old = base_record(str(path.resolve()), text, text)
    old['intake'] = {'mode': 'prepared', 'novelty_check': 'skipped_manual',
                     'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'secret_preflight': 'clear'}
    queue = tmp_path / 'queue/candidates.jsonl'
    append_record(queue, old)
    before = queue.read_bytes()
    assert learn_file(tmp_path, path, client=Client())['queued'] == []
    assert queue.read_bytes() == before
