"""Storage recovery, route isolation and manual-only staging acceptance."""

import json
import sys

import pytest

from glm_rsi_dream.files import read_json
from glm_rsi_dream.predream import absorb as module
from glm_rsi_dream.predream.control import signed_decision
from glm_rsi_dream.predream.queue import append_record, candidate_id, latest_records


def candidate(text, route, status="accepted", pocket=None):
    grade = {"verdict": "accept", "route": route, "pocket": pocket,
             "reason": "Fictional storage fixture", "score": 0.9, "grader": "glm"}
    if status == "rejected":
        grade["verdict"] = "reject"
    elif status == "pending":
        grade = None
    return {"id": candidate_id("fictional-fixture", text), "created_et": "2026-10-12T21:00:00-04:00",
            "source": {"kind": "prompt", "ref": "fictional-fixture"}, "text": text,
            "prompt_evidence": text, "glm_prior": {"answer": "Unknown", "agrees": False},
            "novelty": "new", "status": status, "grade": grade,
            "timings_ms": {"extract": 0, "probe": 0, "grade": 0, "absorb": 0}}


def enqueue(root, *records):
    for record in records:
        if record["status"] == "accepted":
            record["approval"] = signed_decision(root, record, record["grade"], explicit=True)
        append_record(root / "queue" / "candidates.jsonl", record)


def test_routes_preserve_personal_memory_and_separate_training_inputs(tmp_path):
    personal = candidate("Fictional Mara's meeting is tomorrow at 2pm.", "memory")
    weights = candidate("The fictional Orin protocol encodes a checksum in field 4.", "weights")
    pocket = candidate("The fictional Linden fixture uses a Slate-4 console.", "pocket", pocket="fixture")
    pending = candidate("The fictional pending note is unreviewed.", "weights", "pending")
    rejected = candidate("The fictional rejected note is unsupported.", "weights", "rejected")
    enqueue(tmp_path, personal, weights, pocket, pending, rejected)

    result = module.absorb(tmp_path)
    assert result["absorbed"] == [personal["id"]]
    memory = module.read_memory(tmp_path / "memory" / "memory.jsonl")
    assert list(memory) == [personal["id"]]
    assert memory[personal["id"]]["text"] == personal["text"]

    manifest = module.stage(tmp_path, "fixture-run")
    assert manifest["accepted_ids"] == {"weights": [weights["id"]], "pocket": {"fixture": [pocket["id"]]}}
    assert manifest["readiness"] == "raw_facts"
    directory = tmp_path / "staged" / "fixture-run"
    assert read_json(directory / manifest["inputs"][0]["statement_file"]) == {"statement": weights["text"]}
    assert read_json(directory / manifest["inputs"][1]["facts_file"]) == {
        "facts": [{"id": pocket["id"], "statement": pocket["text"]}]}
    snapshots = latest_records(tmp_path / "queue" / "candidates.jsonl")
    assert [row["status"] for row in snapshots] == ["absorbed", "staged", "staged", "pending", "rejected"]
    assert snapshots[0]["timings_ms"]["absorb"] >= 0
    timings = (tmp_path / "timings.jsonl").read_text().splitlines()
    assert [json.loads(row)["stage"] for row in timings] == ["absorb", "stage"]


def test_replay_does_not_duplicate_or_rewrite_approved_inputs(tmp_path):
    personal = candidate("Fictional Mara feels tired today.", "memory")
    pocket = candidate("The fictional rover has two sample bays.", "pocket", pocket="new-pocket")
    enqueue(tmp_path, personal, pocket)
    module.absorb(tmp_path)
    manifest = module.stage(tmp_path, "resume-1")
    memory_bytes = (tmp_path / "memory" / "memory.jsonl").read_bytes()
    manifest_bytes = (tmp_path / "staged" / "resume-1" / "manifest.json").read_bytes()
    queue_bytes = (tmp_path / "queue" / "candidates.jsonl").read_bytes()

    assert module.absorb(tmp_path)["absorbed"] == []
    assert module.stage(tmp_path, "resume-1") == manifest
    assert (tmp_path / "memory" / "memory.jsonl").read_bytes() == memory_bytes
    assert (tmp_path / "staged" / "resume-1" / "manifest.json").read_bytes() == manifest_bytes
    assert (tmp_path / "queue" / "candidates.jsonl").read_bytes() == queue_bytes


def test_resume_after_memory_write_before_queue_receipt(tmp_path, monkeypatch):
    personal = candidate("Fictional Mara's train leaves tonight.", "memory")
    enqueue(tmp_path, personal)

    def interrupted_receipt(path, record):
        raise OSError("Injected interruption before queue receipt")

    monkeypatch.setattr(module, "append_record", interrupted_receipt)
    with pytest.raises(OSError, match="Injected interruption"):
        module.absorb(tmp_path)
    memory_bytes = (tmp_path / "memory" / "memory.jsonl").read_bytes()
    assert latest_records(tmp_path / "queue" / "candidates.jsonl")[0]["status"] == "accepted"

    monkeypatch.setattr(module, "append_record", append_record)
    result = module.absorb(tmp_path)
    assert result["recovered"] == [personal["id"]]
    assert (tmp_path / "memory" / "memory.jsonl").read_bytes() == memory_bytes
    assert latest_records(tmp_path / "queue" / "candidates.jsonl")[0]["status"] == "absorbed"


def test_resume_after_manifest_publish_before_queue_receipt(tmp_path, monkeypatch):
    fact = candidate("The fictional Orin protocol uses four fields.", "weights")
    enqueue(tmp_path, fact)

    def interrupted_receipt(path, record):
        raise OSError("Injected interruption after manifest publish")

    monkeypatch.setattr(module, "append_record", interrupted_receipt)
    with pytest.raises(OSError, match="after manifest publish"):
        module.stage(tmp_path, "interrupted")
    path = tmp_path / "staged" / "interrupted" / "manifest.json"
    manifest_bytes = path.read_bytes()
    monkeypatch.setattr(module, "append_record", append_record)
    module.stage(tmp_path, "interrupted")
    assert path.read_bytes() == manifest_bytes
    assert latest_records(tmp_path / "queue" / "candidates.jsonl")[0]["status"] == "staged"


def test_manual_plan_is_read_only_and_checks_staged_hashes(tmp_path):
    fact = candidate("The fictional Orin protocol includes a checksum.", "weights")
    enqueue(tmp_path, fact)
    manifest = module.stage(tmp_path, "manual")
    before = {}
    for path in tmp_path.rglob("*"):
        if path.is_file():
            before[path] = path.read_bytes()
    before_modules = set(sys.modules)
    plan = module.dream_plan(tmp_path, "manual")
    assert plan["mode"] == "manual"
    assert plan["training_executed"] is False
    assert plan["human_confirm_required"] is True
    assert plan["runs"][0]["readiness"] == "raw_facts"
    assert "glm_rsi_dream.loop" not in set(sys.modules) - before_modules
    for path, contents in before.items():
        assert path.read_bytes() == contents

    statement = tmp_path / "staged" / "manual" / manifest["inputs"][0]["statement_file"]
    statement.write_text('{"statement": "changed"}')
    with pytest.raises(ValueError, match="hash mismatch"):
        module.dream_plan(tmp_path, "manual")


def test_new_facts_require_new_staging_run_and_revocation_blocks_plan(tmp_path):
    first = candidate("The fictional rover uses green status lights.", "weights")
    second = candidate("The fictional rover has six wheels.", "weights")
    enqueue(tmp_path, first)
    module.stage(tmp_path, "sealed")
    enqueue(tmp_path, second)
    with pytest.raises(ValueError, match="sealed"):
        module.stage(tmp_path, "sealed")
    assert module.stage(tmp_path, "new")["accepted_ids"]["weights"] == [second["id"]]
    revoked = dict(first)
    revoked["status"] = "rejected"
    revoked["grade"] = dict(first["grade"], verdict="reject")
    enqueue(tmp_path, revoked)
    with pytest.raises(ValueError, match="no longer approved"):
        module.dream_plan(tmp_path, "sealed")


def test_manual_plan_checks_manifest_metadata_and_empty_absorb_is_timed(tmp_path):
    assert module.absorb(tmp_path)["absorbed"] == []
    assert not (tmp_path / "memory" / "memory.jsonl").exists()
    assert json.loads((tmp_path / "timings.jsonl").read_text())["stage"] == "absorb"
    enqueue(tmp_path, candidate("The fictional rover has two hatches.", "weights"))
    manifest = module.stage(tmp_path, "metadata")
    manifest["inputs"][0]["route"] = "pocket"
    path = tmp_path / "staged" / "metadata" / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        module.dream_plan(tmp_path, "metadata")


@pytest.mark.parametrize("run_id", ["../escape", "/absolute", ".", "..", "nested/run", ""])
def test_stage_rejects_directory_escape(tmp_path, run_id):
    with pytest.raises(ValueError, match="directory name"):
        module.stage(tmp_path, run_id)
    assert not (tmp_path / "staged").exists()
