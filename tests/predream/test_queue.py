"""Queue durability, latest-snapshot views and malformed-state rejection."""

import copy
import json

import pytest

from glm_rsi_dream.predream.queue import (
    append_record,
    candidate_id,
    format_table,
    list_candidates,
)


def record(text="Lumen uses a copper key."):
    return {"id": candidate_id("fixture:lumen", text), "created_et": "2026-10-12T21:50:00-04:00",
            "source": {"kind": "prompt", "ref": "fixture:lumen"}, "text": text,
            "prompt_evidence": text, "glm_prior": {"answer": "Unknown.", "agrees": None},
            "novelty": "new", "status": "pending", "grade": None,
            "timings_ms": {"extract": 5, "probe": 8, "grade": 0, "absorb": 0}}


def test_append_keeps_history_but_list_uses_latest(tmp_path):
    path = tmp_path / "queue" / "candidates.jsonl"
    original = record()
    append_record(path, original)
    original_bytes = path.read_bytes()
    changed = copy.deepcopy(original)
    changed["status"] = "accepted"
    changed["grade"] = {"verdict": "accept", "route": "pocket", "pocket": "fixture",
                        "reason": "Reusable fictional fixture fact.", "score": 0.95, "grader": "glm"}
    append_record(path, changed)
    assert path.read_bytes().startswith(original_bytes)
    assert len(path.read_text().splitlines()) == 2
    assert list_candidates(path) == [changed]
    assert list_candidates(path, "pending") == []
    assert list_candidates(path, "accepted") == [changed]
    table = format_table(list_candidates(path))
    assert original["id"] in table and "accepted" in table and "pocket" in table


@pytest.mark.parametrize("field,value", [
    ("id", "incorrect"), ("created_et", "2026-10-12T21:50:00"),
    ("status", "deleted"), ("novelty", "maybe"), ("prompt_evidence", ""),
    ("glm_prior", {"answer": "Unknown.", "agrees": "false"}),
    ("timings_ms", {"extract": -1, "probe": 0, "grade": 0, "absorb": 0}),
    ("grade", {"verdict": "accept", "route": "weights", "pocket": None,
               "reason": "test", "score": float("nan"), "grader": "glm"}),
])
def test_bad_snapshot_does_not_modify_file(tmp_path, field, value):
    path = tmp_path / "queue.jsonl"
    append_record(path, record())
    before = path.read_bytes()
    invalid = record()
    invalid[field] = value
    with pytest.raises(ValueError):
        append_record(path, invalid)
    assert path.read_bytes() == before


def test_corrupt_tail_is_explicit_failure(tmp_path):
    path = tmp_path / "queue.jsonl"
    append_record(path, record())
    with path.open("a") as handle:
        handle.write('{"unfinished":')
    with pytest.raises(ValueError, match="line 2"):
        list_candidates(path)


def test_missing_queue_and_unknown_filter(tmp_path):
    assert list_candidates(tmp_path / "missing") == []
    assert format_table([]).startswith("ID")
    with pytest.raises(ValueError, match="unknown queue status"):
        list_candidates(tmp_path / "missing", "deleted")


def test_unicode_and_multiline_roundtrip(tmp_path):
    path = tmp_path / "queue.jsonl"
    item = record("Lumen's café\nuses a copper key.")
    append_record(path, item)
    assert len(path.read_text().splitlines()) == 1
    assert json.loads(path.read_text()) == item
