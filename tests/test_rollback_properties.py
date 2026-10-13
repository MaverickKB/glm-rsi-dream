"""Property checks for planned rollback of the original pair."""
import uuid
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from glm_rsi_dream.config import load_config
from glm_rsi_dream.controller import Cluster
from glm_rsi_dream.safety import SafetyError

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = {
    "Config": {"Cmd": ["serve"]},
    "HostConfig": {"Binds": []},
    "Image": "example-image:test",
}


class FakeTransport:
    def __init__(self):
        self.calls = []
        self.running = {}
        self.fail = set()
        self.image = IDENTITY["Image"]

    def call(self, rank, action, **fields):
        container = fields.get("container")
        self.calls.append((rank, action, container))
        if (action, rank) in self.fail or (action, container) in self.fail:
            raise RuntimeError("injected " + str(action))
        if action == "stop":
            self.running[container] = False
            return {}
        if action == "start":
            self.running[container] = True
            return {}
        if action == "inspect":
            return {
                "Config": IDENTITY["Config"],
                "HostConfig": IDENTITY["HostConfig"],
                "Image": self.image,
                "State": {"Running": self.running.get(container, False)},
                "Id": container,
            }
        return {}


def originals():
    rows = []
    for rank in (0, 1):
        rows.append({"id": "original-rank-" + str(rank), "identity": dict(IDENTITY), "site_dir": "/opt/example"})
    return rows


def cluster_for(tmp_path, transport):
    config = load_config(ROOT / "config.example.json")
    config["safety"] = dict(config["safety"])
    run = tmp_path / uuid.uuid4().hex
    cluster = Cluster(config, run, transport)
    cluster.state["originals"] = originals()
    cluster.state["candidates"] = []
    cluster.state["candidate_history"] = []
    cluster.state["trainer_history"] = []
    cluster.health = lambda: None
    cluster.changed = True
    for item in cluster.state["originals"]:
        transport.running[item["id"]] = True
    return cluster


def finish(cluster):
    try:
        cluster._restore()
    except SafetyError as error:
        return str(error)
    return None


def test_successful_restore_starts_worker_then_head(tmp_path):
    transport = FakeTransport()
    cluster = cluster_for(tmp_path, transport)
    assert finish(cluster) is None
    starts = [(rank, container) for rank, action, container in transport.calls if action == "start"]
    assert starts == [(1, "original-rank-1"), (0, "original-rank-0")]
    assert cluster.state["phase"] == "restored"
    assert cluster.state["restore_errors"] == []
    assert cluster.changed is False


def test_malformed_candidate_still_starts_originals(tmp_path):
    transport = FakeTransport()
    cluster = cluster_for(tmp_path, transport)
    cluster.state["candidate_history"] = [{"rank": 0}, {"id": "cand-a"}]
    message = finish(cluster)
    assert message is not None and "candidate record" in message
    starts = [rank for rank, action, _container in transport.calls if action == "start"]
    assert starts == [1, 0]
    assert cluster.state["phase"] == "restore_failed"
    assert cluster.changed is True


def test_duplicate_candidate_is_stopped_once(tmp_path):
    transport = FakeTransport()
    cluster = cluster_for(tmp_path, transport)
    cluster.state["candidate_history"] = [
        {"id": "cand-a", "rank": 0},
        {"id": "cand-a", "rank": 1},
    ]
    assert finish(cluster) is None
    stops = [container for _rank, action, container in transport.calls if action == "stop"]
    assert stops == ["cand-a"]
    assert cluster.changed is False


def test_failed_start_still_attempts_both_ranks(tmp_path):
    transport = FakeTransport()
    transport.fail.add(("start", 1))
    cluster = cluster_for(tmp_path, transport)
    message = finish(cluster)
    assert message is not None and "injected start" in message
    starts = [rank for rank, action, _container in transport.calls if action == "start"]
    assert starts == [1, 0]
    assert cluster.state["phase"] == "restore_failed"
    assert cluster.changed is True


def test_identity_mismatch_stays_visible(tmp_path):
    transport = FakeTransport()
    transport.image = "example-image:other"
    cluster = cluster_for(tmp_path, transport)
    message = finish(cluster)
    assert message is not None and "original exact identity restoration failed" in message
    assert cluster.state["phase"] == "restore_failed"
    assert cluster.changed is True


def test_readiness_timeout_does_not_spin(tmp_path, monkeypatch):
    transport = FakeTransport()
    cluster = cluster_for(tmp_path, transport)
    cluster.config["safety"]["ready_timeout_seconds"] = 0
    monkeypatch.setattr("glm_rsi_dream.controller.time.sleep", lambda _seconds: (_ for _ in ()).throw(AssertionError("slept")))
    message = finish(cluster)
    assert message is not None and "original head readiness failed after restore" in message
    assert cluster.state["phase"] == "restore_failed"


def test_unexpected_health_error_finishes_restore(tmp_path, monkeypatch):
    transport = FakeTransport()
    cluster = cluster_for(tmp_path, transport)

    def fail():
        raise RuntimeError("health probe failed")

    cluster.health = fail
    monkeypatch.setattr("glm_rsi_dream.controller.time.sleep", lambda _seconds: (_ for _ in ()).throw(AssertionError("slept")))
    message = finish(cluster)
    assert message is not None and "health probe failed" in message
    assert cluster.state["phase"] == "restore_failed"
    assert cluster.changed is True


def test_stop_and_inspect_failures_stay_visible(tmp_path, monkeypatch):
    transport = FakeTransport()
    transport.fail.add(("stop", "cand-c"))
    cluster = cluster_for(tmp_path, transport)
    cluster.state["candidate_history"] = [{"id": "cand-c", "rank": 0}]
    message = finish(cluster)
    assert message is not None and "injected stop" in message
    assert cluster.state["phase"] == "restore_failed"
    assert cluster.changed is True

    transport = FakeTransport()
    cluster = cluster_for(tmp_path, transport)
    cluster.state["candidates"] = [{"id": "cand-live", "rank": 0}]
    transport.running["cand-live"] = True
    message = finish(cluster)
    assert message is not None and "candidate remained running" in message

    transport = FakeTransport()
    cluster = cluster_for(tmp_path, transport)

    def fail_inspect(rank, action, **fields):
        if action == "inspect":
            raise RuntimeError("inspect failed")
        return FakeTransport.call(transport, rank, action, **fields)

    transport.call = fail_inspect
    message = finish(cluster)
    assert message is not None and "inspect failed" in message
    assert cluster.state["phase"] == "restore_failed"

    transport = FakeTransport()
    cluster = cluster_for(tmp_path, transport)
    clock = {"now": 0.0}

    def monotonic():
        clock["now"] += 0.4
        return clock["now"]

    def fail_health():
        raise OSError("head down")

    cluster.health = fail_health
    cluster.config["safety"]["ready_timeout_seconds"] = 1
    monkeypatch.setattr("glm_rsi_dream.controller.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("glm_rsi_dream.controller.time.monotonic", monotonic)
    message = finish(cluster)
    assert message is not None and "original head readiness failed after restore" in message


def test_restore_requires_two_originals(tmp_path):
    transport = FakeTransport()
    cluster = cluster_for(tmp_path, transport)
    cluster.state["originals"] = []
    with pytest.raises(SafetyError, match="two preserved"):
        cluster._restore()


def test_candidate_left_running_is_reported(tmp_path):
    transport = FakeTransport()
    cluster = cluster_for(tmp_path, transport)
    cluster.state["candidate_history"] = [{"id": "cand-b", "rank": 1}]
    transport.running["cand-b"] = True

    def call(rank, action, **fields):
        if action == "stop":
            return {}
        return FakeTransport.call(transport, rank, action, **fields)

    transport.call = call
    message = finish(cluster)
    assert message is not None and "remained running" in message
    assert cluster.state["phase"] == "restore_failed"


@given(st.lists(st.one_of(
    st.fixed_dictionaries({
        "id": st.text(alphabet="abc", min_size=1, max_size=6),
        "rank": st.integers(min_value=0, max_value=1),
    }),
    st.fixed_dictionaries({"rank": st.integers(min_value=0, max_value=1)}),
    st.fixed_dictionaries({"id": st.text(alphabet="abc", min_size=1, max_size=4)}),
    st.fixed_dictionaries({
        "id": st.lists(st.integers(min_value=0, max_value=1), max_size=2),
        "rank": st.just(0),
    }),
    st.none(),
    st.integers(min_value=0, max_value=3),
), max_size=5))
def test_restore_always_finishes_and_starts_both_originals(tmp_path, history):
    transport = FakeTransport()
    cluster = cluster_for(tmp_path, transport)
    cluster.state["candidate_history"] = history
    message = finish(cluster)
    starts = [rank for rank, action, _container in transport.calls if action == "start"]
    assert starts == [1, 0]
    assert cluster.state["phase"] in ("restored", "restore_failed")
    well_formed = all(
        isinstance(item, dict) and isinstance(item.get("id"), str) and type(item.get("rank")) is int
        for item in history
    )
    if well_formed:
        assert message is None
        assert cluster.state["phase"] == "restored"
        assert cluster.changed is False
    else:
        assert message is not None
        assert cluster.state["phase"] == "restore_failed"
        assert cluster.changed is True
