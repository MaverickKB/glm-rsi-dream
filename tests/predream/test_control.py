import json

import pytest

from glm_rsi_dream.predream import control
from glm_rsi_dream.predream.absorb import absorb, stage
from glm_rsi_dream.predream.policy import record_policy
from glm_rsi_dream.predream.queue import append_record, candidate_id, latest_records


def configure(tmp_path, monkeypatch, autonomy="manual", allowed=None):
    root = tmp_path / "run"
    key = tmp_path / "admin.key"
    permission = tmp_path / "permission.json"
    control.initialize_key(key)
    monkeypatch.setenv("PREDREAM_ADMIN_KEY_FILE", str(key))
    monkeypatch.setenv("PREDREAM_MODE", "harness")
    monkeypatch.delenv("PREDREAM_AUTONOMY", raising=False)
    monkeypatch.delenv("PREDREAM_ADMIN_PERMISSION_FILE", raising=False)
    control.issue_permission(permission, admin_id="test-owner", root_scope=root,
                             modes=["harness"], autonomy=allowed or [autonomy])
    policy = {"schema_version": 1, "name": "gate-test", "extract": {}, "grade": {},
              "control": {"mode": "harness", "autonomy": autonomy, "permission_file": str(permission)}}
    record_policy(root, policy)
    return root, permission


def proposed(route="memory"):
    text = "Fictional Lee's appointment is tomorrow."
    row = {"id": candidate_id("gate-fixture", text), "created_et": "2026-10-12T22:30:00-04:00",
           "source": {"kind": "prompt", "ref": "gate-fixture"}, "text": text,
           "prompt_evidence": text, "glm_prior": {"answer": "Unknown", "agrees": None},
           "novelty": "new", "status": "pending", "grade": None,
           "timings_ms": {"extract": 0, "probe": 0, "grade": 0, "absorb": 0}}
    grade = {"verdict": "accept", "route": route, "pocket": None,
             "reason": "Explicit supported fixture fact.", "score": 0.95, "grader": "glm"}
    return row, grade


def test_manual_waits_then_explicit_admin_opens_handoff(tmp_path, monkeypatch):
    root, _ = configure(tmp_path, monkeypatch)
    row, grade = proposed()
    queued = control.apply_proposal(root, row, grade)
    assert queued["status"] == "pending" and queued["grade"] is None
    append_record(root / "queue/candidates.jsonl", queued)
    assert absorb(root)["absorbed"] == []
    assert stage(root, "manual")["items"] == []
    assert control.approve(root, [row["id"]])["approved"] == [row["id"]]
    approved = latest_records(root / "queue/candidates.jsonl")[0]
    assert approved["approval"]["decision"]["explicit"] is True
    assert absorb(root)["absorbed"] == [row["id"]]


def test_automatic_permission_cannot_be_self_upgraded_or_forged(tmp_path, monkeypatch):
    root, permission = configure(tmp_path, monkeypatch, autonomy="automatic", allowed=["manual"])
    row, grade = proposed()
    result = control.apply_proposal(root, row, grade)
    assert result["status"] == "pending"
    assert "does not authorize" in result["proposal"]["reason"]
    grant = json.loads(permission.read_text())
    grant["permission"]["autonomy"].append("automatic")
    permission.write_text(json.dumps(grant))
    result = control.apply_proposal(root, row, grade)
    assert result["status"] == "pending"
    assert "signature is invalid" in result["proposal"]["reason"]


def test_unapproved_or_changed_route_cannot_stage(tmp_path, monkeypatch):
    root, _ = configure(tmp_path, monkeypatch, autonomy="automatic")
    row, grade = proposed("weights")
    unapproved = dict(row, status="accepted", grade=grade)
    append_record(root / "queue/candidates.jsonl", unapproved)
    with pytest.raises(ValueError, match="admin-approved"):
        stage(root, "unapproved")
    approved = control.apply_proposal(root, row, grade)
    approved["grade"] = dict(grade, route="memory")
    append_record(root / "queue/candidates.jsonl", approved)
    with pytest.raises(ValueError, match="changed after"):
        absorb(root)


def test_supervised_opens_only_configured_high_confidence_routes(tmp_path, monkeypatch):
    root, _ = configure(tmp_path, monkeypatch, autonomy="supervised")
    row, grade = proposed()
    assert control.apply_proposal(root, row, grade)["status"] == "accepted"
    assert control.apply_proposal(root, row, dict(grade, score=0.2))["status"] == "pending"
    assert control.apply_proposal(root, row, dict(grade, route="weights"))["status"] == "pending"
    row["novelty"] = "conflicts"
    assert control.apply_proposal(root, row, grade)["status"] == "pending"


def test_missing_permission_preserves_model_proposal(tmp_path, monkeypatch):
    for name in ("PREDREAM_ADMIN_KEY_FILE", "PREDREAM_ADMIN_PERMISSION_FILE", "PREDREAM_AUTONOMY", "PREDREAM_MODE"):
        monkeypatch.delenv(name, raising=False)
    row, grade = proposed()
    result = control.apply_proposal(tmp_path, row, grade)
    assert result["status"] == "pending" and result["proposal"]["grade"] == grade
    with pytest.raises(ValueError, match="permission is unavailable"):
        control.approve(tmp_path, [row["id"]])


def test_invalid_admin_batch_has_no_partial_approval(tmp_path, monkeypatch):
    root, _ = configure(tmp_path, monkeypatch)
    row, grade = proposed()
    append_record(root / "queue/candidates.jsonl", control.apply_proposal(root, row, grade))
    before = (root / "queue/candidates.jsonl").read_bytes()
    with pytest.raises(ValueError, match="existing pending"):
        control.approve(root, [row["id"], "absent"])
    assert (root / "queue/candidates.jsonl").read_bytes() == before
