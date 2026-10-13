import json

import pytest

from glm_rsi_dream.predream import timing


def test_stage_timer_records_et_and_propagates_failure(tmp_path, monkeypatch):
    clock = iter([10, 10.25])
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(clock))
    with pytest.raises(RuntimeError, match="model unavailable"):
        with timing.StageTimer(tmp_path, "grade", "candidate") as measured:
            raise RuntimeError("model unavailable")
    record = json.loads((tmp_path / "timings.jsonl").read_text())
    assert record["wall_ms"] == 250
    assert measured.wall_ms == 250
    assert record["start_et"].endswith(("-04:00", "-05:00"))
    assert record["error"] == "model unavailable"


def test_summary_counts_only_successes_and_keeps_appended_records(tmp_path):
    timing.record_timing(tmp_path, "grade", "a", "2026-10-12T21:00:00-04:00", 10)
    timing.record_timing(tmp_path, "grade", "b", "2026-10-12T21:00:01-04:00", 30)
    timing.record_timing(tmp_path, "grade", "c", "2026-10-12T21:00:02-04:00", 100, error="failed")
    summary = timing.summarize_timings(tmp_path / "timings.jsonl")
    assert summary["grade"] == {"count": 2, "average_ms": 20, "failed_count": 1}
    assert len((tmp_path / "timings.jsonl").read_text().splitlines()) == 3


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf")])
def test_invalid_duration_does_not_write(tmp_path, duration):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        timing.record_timing(tmp_path, "probe", "a", timing.now_et(), duration)
    assert not (tmp_path / "timings.jsonl").exists()
