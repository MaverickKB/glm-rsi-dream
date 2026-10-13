"""Recorded GLM exchange playback, distinct from synthetic client tripwires."""

import json
import socket
import subprocess
from pathlib import Path

import pytest

from glm_rsi_dream.predream.queue import latest_records
from glm_rsi_dream.predream.tap_policy import TapPolicy
from glm_rsi_dream.predream.tap_replay import RecordedClient, replay

TRACE = Path(__file__).parent / "fixtures/tap_recorded.jsonl"


@pytest.fixture(autouse=True)
def no_external_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("recorded replay must not launch a process or network connection")
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


@pytest.mark.parametrize("mode", ["spark", "harness"])
def test_actual_recorded_glm_answers_enter_pending_queue_without_generation(tmp_path, mode):
    policy = TapPolicy(enabled=True, mode=mode, endpoint="http://recorded.invalid:8888",
                       principals=["recorded-fixture"])
    result = replay(TRACE, tmp_path, policy=policy, principal="recorded-fixture", opt_in=True)
    assert result["observed"] == {"captured": 1}
    assert result["processed"] == 1 and result["queued"] == 4 and result["model_calls"] == 9
    rows = latest_records(tmp_path / "queue/candidates.jsonl")
    assert len(rows) == 4
    for row in rows:
        assert row["novelty"] == "known" and row["status"] == "pending"
        assert row["grade"] is None
    repeated = replay(TRACE, tmp_path, policy=policy, principal="recorded-fixture", opt_in=True)
    assert repeated["observed"] == {"duplicate": 1} and repeated["model_calls"] == 0
    assert len((tmp_path / "queue/candidates.jsonl").read_text().splitlines()) == 4


def test_disabled_and_unconsented_replay_do_not_read_missing_traffic(tmp_path):
    root = tmp_path / "absent"
    assert replay("missing", root, policy=TapPolicy(), principal="fixture")["status"] == "disabled"
    enabled = TapPolicy(enabled=True, endpoint="http://recorded.invalid:8888", principals=["fixture"])
    assert replay("missing", root, policy=enabled, principal="fixture")["status"] == "no_consent"
    assert not root.exists()


def test_missing_exchange_has_no_invented_response():
    records = []
    for line in TRACE.read_text().splitlines():
        records.append(json.loads(line))
    client = RecordedClient(records)
    messages = records[0]["messages"][:1] + [{"role": "user", "content": "unrecorded input"}]
    with pytest.raises(ValueError, match="No exact recorded exchange"):
        client.chat(messages)
