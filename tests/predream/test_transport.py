"""Keep planned startup outages distinct from semantic and broken-rank failures."""

import errno
import json
import urllib.error
from email.message import Message

import pytest

from glm_rsi_dream.predream import extract


def test_startup_503_retries_same_request_and_preserves_event(tmp_path, monkeypatch):
    clock = [0]
    requests = []
    client = extract.GLMClient("http://engine:8888", "glm", timeout=10,
                               trace_path=tmp_path / "traces/glm.jsonl")

    def request(messages, *, temperature=None):
        requests.append((messages, temperature))
        if len(requests) == 1:
            raise urllib.error.HTTPError("http://engine:8888", 503, "starting", {}, None)
        return "actual generation reply"

    def sleep(seconds):
        clock[0] += seconds

    monkeypatch.setattr(client, "_chat_once", request)
    monkeypatch.setattr(extract.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(extract.time, "sleep", sleep)
    messages = [{"role": "user", "content": "question"}]
    assert client.chat(messages, temperature=0.7) == "actual generation reply"
    assert requests == [(messages, 0.7), (messages, 0.7)]
    event = json.loads((tmp_path / "traces/transport-events.jsonl").read_text())
    assert event["attempt"] == 1 and event["wait_seconds"] == 5


@pytest.mark.parametrize("error", [
    urllib.error.HTTPError("http://engine:8888", 500, "ranks out of step", Message(), None),
    urllib.error.URLError(OSError(errno.EHOSTUNREACH, "No route to host")),
])
def test_broken_rank_and_host_access_fail_without_transport_retry(error, monkeypatch):
    client = extract.GLMClient("http://engine:8888", "glm")
    calls = []

    def request(messages, **options):
        calls.append(messages)
        raise error

    monkeypatch.setattr(client, "_chat_once", request)
    with pytest.raises(urllib.error.URLError):
        client.chat([{"role": "user", "content": "question"}])
    assert len(calls) == 1


def test_connection_refusal_has_bounded_wait_and_never_invents_reply(monkeypatch):
    client = extract.GLMClient("http://engine:8888", "glm", timeout=6)
    clock = [0]
    calls = []

    def request(messages, **options):
        calls.append(messages)
        raise urllib.error.URLError(OSError(errno.ECONNREFUSED, "not started"))

    def sleep(seconds):
        clock[0] += seconds

    monkeypatch.setattr(client, "_chat_once", request)
    monkeypatch.setattr(extract.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(extract.time, "sleep", sleep)
    with pytest.raises(urllib.error.URLError):
        client.chat([{"role": "user", "content": "question"}])
    assert clock[0] == 6 and len(calls) == 3
