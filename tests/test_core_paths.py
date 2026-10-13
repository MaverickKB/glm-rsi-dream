"""CPU coverage for admission boundaries, routing, and rollback helpers."""
import importlib.machinery
import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from glm_rsi_dream.dataset import admitted
from glm_rsi_dream.grading import decision, parse_verdict
from glm_rsi_dream.safety import MemoryGate, SafetyError

WORD = st.text(alphabet="abcxyz", min_size=1, max_size=6)


@given(WORD, WORD)
def test_admission_requires_every_group(left, right):
    routes = [{"all": [[left], [right]]}]
    assert admitted(left, routes) is (left == right)
    assert admitted(left + " " + right, routes) is True
    assert admitted("xx" + left + "yy" + right + "zz", routes) is False
    assert admitted("", []) is False


@given(
    available=st.integers(min_value=-5, max_value=5_000),
    floor=st.integers(min_value=0, max_value=5_000),
)
def test_memory_floor_follows_the_sample(available, floor):
    gate = MemoryGate({
        "memory_floor_bytes": floor,
        "swap_out_bytes_per_second": 10**9,
        "swap_sustain_seconds": 30,
    })
    sample = {"available": available, "time": 1.0, "swap_out_bytes": 0}
    if available < 0:
        with pytest.raises(SafetyError, match="invalid"):
            gate.observe(0, sample)
        return
    if available < floor:
        with pytest.raises(SafetyError, match="memory floor"):
            gate.observe(0, sample)
        return
    gate.observe(0, sample)
    assert gate.minima[0] == available


def test_swap_gate_rejects_a_sustained_burst_and_bad_clocks():
    gate = MemoryGate({
        "memory_floor_bytes": 1,
        "swap_out_bytes_per_second": 10,
        "swap_sustain_seconds": 2,
    })
    gate.observe(0, {"available": 100, "time": 1.0, "swap_out_bytes": 0})
    with pytest.raises(SafetyError, match="invalid"):
        gate.observe(0, {"available": 100, "time": 1.0, "swap_out_bytes": 0})
    gate.observe(0, {"available": 100, "time": 2.0, "swap_out_bytes": 100})
    with pytest.raises(SafetyError, match="sustained swap-out"):
        gate.observe(0, {"available": 100, "time": 4.0, "swap_out_bytes": 200})
    with pytest.raises(SafetyError, match="invalid"):
        gate.observe(1, {"available": float("nan"), "time": 1.0, "swap_out_bytes": 0})


@given(st.booleans(), st.booleans(), st.text(alphabet="abc", min_size=1, max_size=12))
def test_verdict_requires_exact_boolean_fields(correct, grounded, reason):
    raw = json.dumps({"correct": correct, "grounded": grounded, "reason": reason})
    parsed = parse_verdict(raw)
    assert parsed == {"correct": correct, "grounded": grounded, "reason": reason}
    extra = json.loads(raw)
    extra["note"] = "extra"
    with pytest.raises(SafetyError, match="fields"):
        parse_verdict(json.dumps(extra))


def test_decision_keeps_only_with_gain_and_intact_controls():
    rows = [
        {"fact_id": "alpha", "memory": {"grounded": True}, "baseline": {"grounded": False}},
        {"fact_id": "beta", "memory": {"grounded": True}, "baseline": {"grounded": False}},
    ]
    acceptance = {"minimum_grounded_recall": 0.5, "minimum_gain": 0.2, "require_per_fact": True}
    controls = [{"byte_identical": True, "zero_installation": True,
                 "zero_cache_writes": True, "zero_offpath_delta": True}]
    assert decision(rows, controls, acceptance)["verdict"] == "keep"
    controls[0]["zero_installation"] = False
    assert decision(rows, controls, acceptance)["verdict"] == "reject"


def test_overlay_does_not_replace_a_loader_without_a_path(monkeypatch):
    from tensorfold_dream.bootstrap import OVERLAY_MODULES, OverlayFinder
    from tensorfold_dream.capture_native import PROCESS_CAPTURE

    fullname = next(iter(OVERLAY_MODULES))
    loader = importlib.machinery.SourceFileLoader(fullname, "/tmp/tensorfold-example.py")
    spec = importlib.machinery.ModuleSpec(fullname, loader, origin=None)
    saved = dict(PROCESS_CAPTURE)
    monkeypatch.setattr(
        importlib.machinery.PathFinder,
        "find_spec",
        staticmethod(lambda name, path=None: spec if name == fullname else None),
    )
    try:
        found = OverlayFinder().find_spec(fullname)
    finally:
        PROCESS_CAPTURE.clear()
        PROCESS_CAPTURE.update(saved)
    assert found is spec
    assert found.loader is loader
