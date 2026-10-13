"""Append stage wall times and summarize successful calls."""

import fcntl
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def now_et() -> Any:
    return datetime.now(ET).isoformat()


def record_timing(root: Any, stage: Any, candidate_id: Any, start_et: Any, wall_ms: Any, error: Any = None) -> Any:
    duration = float(wall_ms)
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("wall_ms must be finite and nonnegative")
    timestamp = datetime.fromisoformat(start_et)
    if timestamp.utcoffset() is None:
        raise ValueError("start_et must include a timezone")
    record = {
        "stage": stage,
        "id": candidate_id,
        "start_et": timestamp.astimezone(ET).isoformat(),
        "wall_ms": duration,
    }
    if error is not None:
        record["error"] = str(error)
    path = Path(root) / "timings.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return record


class StageTimer:
    def __init__(self, root: Any, stage: Any, candidate_id: Any) -> None:
        self.root = root
        self.stage = stage
        self.candidate_id = candidate_id
        self.wall_ms: float | None = None

    def __enter__(self) -> Any:
        self.start_et = now_et()
        self.started = time.perf_counter()
        return self

    def __exit__(self, error_type: Any, error: Any, traceback: Any) -> Any:
        self.wall_ms = (time.perf_counter() - self.started) * 1000
        record_timing(
            self.root, self.stage, self.candidate_id,
            self.start_et, self.wall_ms, error=error,
        )
        return False


def summarize_timings(path: Any) -> Any:
    totals = {}
    counts = {}
    failures = {}
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            stage = record["stage"]
            if stage not in totals:
                totals[stage] = 0.0
                counts[stage] = 0
                failures[stage] = 0
            if "error" in record:
                failures[stage] += 1
                continue
            totals[stage] += record["wall_ms"]
            counts[stage] += 1
    summary = {}
    for stage in sorted(totals):
        average = None
        if counts[stage]:
            average = totals[stage] / counts[stage]
        summary[stage] = {
            "count": counts[stage],
            "average_ms": average,
            "failed_count": failures[stage],
        }
    return summary
