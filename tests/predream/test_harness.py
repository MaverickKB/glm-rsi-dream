import json
from pathlib import Path

import pytest

from glm_rsi_dream.predream import control
from glm_rsi_dream.predream.harness import handoff, prepare, run_funnel
from glm_rsi_dream.predream.queue import append_record, candidate_id, latest_records
from glm_rsi_dream.predream.timing import now_et


def config(tmp_path, autonomy="manual", mode="harness", permission=None):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "schema_version": 1, "name": "test", "extract": {}, "grade": {},
        "control": {"mode": mode, "autonomy": autonomy, "permission_file": permission},
    }))
    return path


def candidate():
    text = "The fictional lamp is on shelf A."
    return {
        "id": candidate_id("fictional", text), "created_et": now_et(),
        "source": {"kind": "prompt", "ref": "fictional"},
        "text": text,
        "prompt_evidence": "The fictional lamp is on shelf A.",
        "glm_prior": {"answer": "I do not know.", "agrees": None},
        "novelty": "new", "status": "pending", "grade": None,
        "timings_ms": {"extract": 0, "probe": 0, "grade": 0, "absorb": 0},
    }


GRADE = {"verdict": "accept", "route": "memory", "pocket": None,
         "reason": "Source supports a temporary location", "score": 0.95, "grader": "glm"}


def test_manual_proposal_has_no_retention_then_explicit_controller_approval_opens_handoff(tmp_path, monkeypatch):
    root = tmp_path / "run"
    key = tmp_path / "admin.key"
    control.initialize_key(key)
    monkeypatch.setenv("PREDREAM_ADMIN_KEY_FILE", str(key))
    grant = tmp_path / "grant.json"
    control.issue_permission(grant, admin_id="test-admin", root_scope=root,
                             modes=["harness"], autonomy=["manual"])
    prepare(root, config(tmp_path, permission=str(grant)))
    proposed = control.apply_proposal(root, candidate(), GRADE)
    append_record(root / "queue/candidates.jsonl", proposed)
    result = handoff(root, "batch")
    assert result["accepted_ready"] == []
    assert not (root / "memory").exists()
    assert not (root / "staged").exists()
    control.approve(root, [proposed["id"]])
    handoff(root, "batch")
    assert latest_records(root / "queue/candidates.jsonl")[0]["status"] == "absorbed"
    assert (root / "memory/memory.jsonl").exists()


def test_unsigned_accept_cannot_reach_local_handoff(tmp_path):
    root = tmp_path / "run"
    prepare(root, config(tmp_path))
    record = candidate()
    record.update(status="accepted", grade=GRADE)
    append_record(root / "queue/candidates.jsonl", record)
    with pytest.raises(ValueError, match="Admin permission"):
        handoff(root, "batch")
    assert not (root / "memory").exists()
    assert not (root / "staged").exists()


def test_spark_mode_is_rejected_by_external_entrypoint(tmp_path, monkeypatch):
    monkeypatch.setenv("PREDREAM_MODE", "spark")
    with pytest.raises(ValueError, match="mode=harness"):
        prepare(tmp_path / "run", config(tmp_path, mode="spark"))


def test_non_generation_destination_is_rejected_before_source_or_run_access(tmp_path):
    with pytest.raises(ValueError, match="port8888"):
        run_funnel(Path("missing"), tmp_path / "run", policy_path=Path("missing"),
                   endpoint="http://example.invalid:9000", run_id="batch")
    assert not (tmp_path / "run").exists()
