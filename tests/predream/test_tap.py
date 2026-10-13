"""Consent, privacy, budgets and restart tripwires for real queue integration."""

import copy
import json
import socket
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from glm_rsi_dream.predream import tap as module
from glm_rsi_dream.predream.queue import latest_records
from glm_rsi_dream.predream.tap import TrafficTap, drain, observe_completion
from glm_rsi_dream.predream.tap_cli import main
from glm_rsi_dream.predream.tap_policy import TapPolicy
from glm_rsi_dream.predream.tap_store import database, revoke

ENDPOINT = "http://recorded.invalid:8888"


def policy(**changes):
    settings = {"enabled": True, "principals": ["fixture"], "endpoint": ENDPOINT}
    settings.update(changes)
    return TapPolicy(**settings)


def observe(tap, text="The fictional workshop has cobalt washers.", **changes):
    context = {"principal": "fixture", "request_id": "one", "opt_in": True,
               "mode": tap.policy.mode, "endpoint": ENDPOINT}
    context.update(changes)
    request = {"messages": [{"role": "user", "content": text}]}
    return tap.observe(request, **context)


class FixtureClient:
    def __init__(self, answer=None):
        self.calls = 0
        self.answer = answer

    def chat(self, messages, **kwargs):
        self.calls += 1
        if self.answer is not None:
            return self.answer
        system = messages[0]["content"]
        if system.startswith("Extract atomic"):
            text = messages[1]["content"]
            return json.dumps({"candidates": [{"text": text, "prompt_evidence": text,
                                               "probe_question": "What does the fictional workshop contain?"}]})
        if system.startswith("Answer the question"):
            return "I do not know."
        return '{"agrees":null,"novelty":"new","reason":"unknown"}'


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("tap tests must not open a network connection")
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def test_fully_disabled_has_no_effect_even_with_missing_inputs(tmp_path):
    root = tmp_path / "absent"
    tap = TrafficTap(root)
    assert observe(tap, object())["status"] == "disabled"
    assert drain(root, policy=TapPolicy(), client=None)["status"] == "disabled"
    assert main(["--root", str(root), "replay", "missing.jsonl", "--principal", "fixture"]) == 0
    assert not root.exists()


@pytest.mark.parametrize("changes,status", [
    ({"opt_in": False}, "no_consent"), ({"opt_in": 1}, "no_consent"),
    ({"principal": "other"}, "no_consent"), ({"mode": "spark"}, "ineligible"),
    ({"successful": False}, "ineligible"), ({"internal": True}, "ineligible"),
    ({"endpoint": "http://worker.invalid:8888"}, "endpoint"),
    ({"endpoint": "http://recorded.invalid:8000"}, "endpoint"),
    ({"request_id": ""}, "request_id"),
])
def test_trusted_context_gates_create_no_storage(tmp_path, changes, status):
    root = tmp_path / "tap"
    assert observe(TrafficTap(root, policy()), **changes)["status"] == status
    assert not root.exists()


@pytest.mark.parametrize("text", [
    "My password is hunter-two.", "Send Authorization: Bearer abcdef.",
    "hf_abcdefghijklmnopqrstuvwxyz", "email person@example.test",
    "call +1 (202) 555-0123", "SSN 123-45-6789", "credit card 4111111111111111",
    "Patient diagnosis is private.", "The address is 12 Fictional Street.",
    "Do not remember the workshop contents.", "NEVER SAVE this note.",
    "-----BEGIN RSA PRIVATE KEY-----", "api_key=abcd", "secret=abcd",
])
def test_private_text_never_reaches_disk(tmp_path, text):
    root = tmp_path / "tap"
    assert observe(TrafficTap(root, policy()), text)["status"] == "private_content"
    assert not root.exists()


def test_deny_terms_size_multimodal_and_history(tmp_path):
    tap = TrafficTap(tmp_path, policy(denied_terms=["restricted-project"]))
    assert observe(tap, "restricted-project is green")["status"] == "denied_term"
    assert observe(tap, "x" * 8193)["status"] == "oversize"
    assert observe(tap, [{"type": "image"}])["status"] == "unsupported"
    request = {"metadata": {"opt_in": True}, "messages": [
        {"role": "system", "content": "password system-secret"},
        {"role": "assistant", "content": "secret assistant-answer"},
        {"role": "user", "content": "The fictional workshop has cobalt washers."}]}
    original = copy.deepcopy(request)
    response = {"choices": [{"message": {"content": "secret response-content"}}]}
    output = observe_completion(tap, request, response, principal="fixture", request_id="final",
                                opt_in=True, mode="harness", endpoint=ENDPOINT)
    assert output is response and request == original
    assert drain(tmp_path, policy=tap.policy, client=FixtureClient())["queued"] == 1
    for path in tmp_path.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            assert b"system-secret" not in data and b"assistant-answer" not in data
            assert b"response-content" not in data


@pytest.mark.parametrize("mode", ["spark", "harness"])
def test_mode_callbacks_feed_existing_pending_queue(tmp_path, mode):
    settings = policy(mode=mode)
    tap = TrafficTap(tmp_path, settings)
    callback = tap.spark_completed if mode == "spark" else tap.harness_completed
    request = {"messages": [{"role": "user", "content": "The fictional workshop has cobalt washers."}]}
    assert callback(request, principal="fixture", request_id="one", opt_in=True,
                    endpoint=ENDPOINT)["status"] == "captured"
    client = FixtureClient()
    assert drain(tmp_path, policy=settings, client=client) == {"status": "ok", "processed": 1, "queued": 1}
    rows = latest_records(tmp_path / "queue/candidates.jsonl")
    assert len(rows) == 1 and rows[0]["status"] == "pending" and rows[0]["grade"] is None
    assert rows[0]["novelty"] == "new" and rows[0]["source"]["kind"] == "conversation"
    assert client.calls == 3
    assert observe(TrafficTap(tmp_path, settings))["status"] == "duplicate"
    assert drain(tmp_path, policy=settings, client=client)["processed"] == 0
    assert client.calls == 3
    assert (tmp_path / "tap").stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "tap/spool.sqlite3").stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "queue/candidates.jsonl").stat().st_mode & 0o777 == 0o600
    assert not (tmp_path / "traces").exists()


def test_persisted_sliding_rates_and_capacity(tmp_path):
    clock = [1000.0]
    settings = policy(events_per_minute=2, principal_events_per_minute=1,
                      principals=["fixture", "second"])
    tap = TrafficTap(tmp_path, settings, clock=lambda: clock[0])
    assert observe(tap)["status"] == "captured"
    assert observe(tap, request_id="two")["status"] == "rate_limited"
    assert observe(tap, request_id="two", principal="second")["status"] == "captured"
    restarted = TrafficTap(tmp_path, settings, clock=lambda: clock[0])
    assert observe(restarted, request_id="three", principal="second")["status"] == "rate_limited"
    clock[0] += 61
    assert observe(restarted, request_id="three")["status"] == "captured"
    assert observe(TrafficTap(tmp_path, replace(settings, max_events=3), clock=lambda: clock[0]),
                   request_id="four")["status"] == "capacity"


def test_bytes_budget_and_expiry(tmp_path):
    settings = policy(max_spool_bytes=20)
    tap = TrafficTap(tmp_path, settings)
    assert observe(tap)["status"] == "capacity"
    settings = policy(retention_seconds=60)
    tap = TrafficTap(tmp_path, settings, clock=lambda: 1000)
    assert observe(tap)["status"] == "captured"
    assert drain(tmp_path, policy=settings, client=FixtureClient(), clock=lambda: 1061)["processed"] == 0
    assert observe(TrafficTap(tmp_path, settings, clock=lambda: 1061))["status"] == "captured"


def test_revocation_and_allowlist_removal_prevent_model_access(tmp_path):
    settings = policy()
    tap = TrafficTap(tmp_path, settings)
    observe(tap)
    assert revoke(tmp_path, "fixture")["purged_events"] == 1
    assert observe(tap, request_id="two")["status"] == "revoked"
    assert drain(tmp_path, policy=settings, client=FixtureClient())["processed"] == 0
    other = tmp_path / "other"
    observe(TrafficTap(other, settings))
    client = FixtureClient()
    assert drain(other, policy=replace(settings, principals=[]), client=client)["queued"] == 0
    assert client.calls == 0


def test_model_budget_survives_restart(tmp_path):
    settings = policy(model_calls_per_minute=1)
    observe(TrafficTap(tmp_path, settings))
    client = FixtureClient()
    assert drain(tmp_path, policy=settings, client=client)["status"] == "model_rate_limited"
    assert client.calls == 1
    assert drain(tmp_path, policy=settings, client=client)["status"] == "model_rate_limited"
    assert client.calls == 1 and latest_records(tmp_path / "queue/candidates.jsonl") == []


@pytest.mark.parametrize("answer", ["password leaked-value", "{broken", "x" * 9000])
def test_model_errors_and_private_outputs_are_never_persisted(tmp_path, answer):
    settings = policy()
    observe(TrafficTap(tmp_path, settings))
    client = FixtureClient(answer)
    assert drain(tmp_path, policy=settings, client=client)["status"] == "error"
    assert latest_records(tmp_path / "queue/candidates.jsonl") == []
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert answer.encode() not in path.read_bytes()


def test_queue_capacity_preserves_ready_batch_without_more_model_calls(tmp_path):
    settings = policy(principal_events_per_minute=10, pending_candidates=1)
    tap = TrafficTap(tmp_path, settings)
    observe(tap)
    observe(tap, "The fictional workshop holds amber screws.", request_id="two")
    client = FixtureClient()
    result = drain(tmp_path, policy=settings, client=client)
    assert result == {"status": "queue_capacity", "processed": 1, "queued": 1}
    assert client.calls == 6
    result = drain(tmp_path, policy=replace(settings, pending_candidates=2), client=client)
    assert result["queued"] == 1 and client.calls == 6


def test_interrupt_after_queue_append_resumes_exactly_once(tmp_path, monkeypatch):
    settings = policy()
    observe(TrafficTap(tmp_path, settings))
    real_append = module.append_record

    def append_then_interrupt(path, record):
        real_append(path, record)
        raise KeyboardInterrupt()
    monkeypatch.setattr(module, "append_record", append_then_interrupt)
    client = FixtureClient()
    with pytest.raises(KeyboardInterrupt):
        drain(tmp_path, policy=settings, client=client)
    monkeypatch.setattr(module, "append_record", real_append)
    assert drain(tmp_path, policy=settings, client=client)["processed"] == 1
    assert client.calls == 3 and len(latest_records(tmp_path / "queue/candidates.jsonl")) == 1
    assert len((tmp_path / "queue/candidates.jsonl").read_text().splitlines()) == 1


def test_parallel_capture_is_bounded_and_callback_failure_is_soft(tmp_path):
    settings = policy(max_events=2, events_per_minute=20, principal_events_per_minute=20)
    observe(TrafficTap(tmp_path, settings))

    def submit(index):
        return observe(TrafficTap(tmp_path, settings), request_id=str(index))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(submit, range(8)))
    with database(tmp_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 2
    assert sum(row["status"] == "captured" for row in results) == 1
    broken = tmp_path / "file"
    broken.write_text("not a directory")
    assert observe(TrafficTap(broken, settings))["status"] == "unavailable"


@pytest.mark.parametrize("values", [{"enabled": "false"}, {"max_events": 0},
                                    {"events_per_minute": True}, {"retention_seconds": 1},
                                    {"enabled": True}, {"mode": "worker"}])
def test_policy_rejects_ambiguous_configuration(values):
    with pytest.raises(ValueError):
        TapPolicy(**values)


def test_running_callback_reloads_off_switch_and_invalid_policy(tmp_path):
    configuration = tmp_path / "policy.json"
    from dataclasses import asdict
    configuration.write_text(json.dumps(asdict(policy())))
    root = tmp_path / "spool"
    tap = TrafficTap(root, policy_path=configuration)
    assert observe(tap)["status"] == "captured"
    configuration.write_text(json.dumps({"enabled": False}))
    assert observe(tap, request_id="two")["status"] == "disabled"
    configuration.write_text("broken")
    assert observe(tap, request_id="three")["status"] == "unavailable"
    with database(root) as connection:
        assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


def test_collector_rejects_raw_trace_client(tmp_path):
    settings = policy()
    observe(TrafficTap(tmp_path, settings))
    client = FixtureClient()
    client.trace_path = tmp_path / "raw.jsonl"
    with pytest.raises(ValueError, match="raw tracing disabled"):
        drain(tmp_path, policy=settings, client=client)
    assert client.calls == 0


def test_spool_cannot_cross_mode_or_endpoint_boundary(tmp_path):
    settings = policy()
    observe(TrafficTap(tmp_path, settings))
    client = FixtureClient()
    for changed in (replace(settings, mode="spark"),
                    replace(settings, endpoint="http://different.invalid:8888")):
        with pytest.raises(ValueError, match="use a new private root"):
            drain(tmp_path, policy=changed, client=client)
    assert client.calls == 0


def test_transport_failure_consumes_one_call_without_hidden_retries(tmp_path, monkeypatch):
    from glm_rsi_dream.predream.extract import GLMClient
    from glm_rsi_dream.predream.tap import TapGLMClient
    attempts = []

    def fail_once(self, messages, **kwargs):
        attempts.append(messages)
        raise OSError("fixture transport failure")
    monkeypatch.setattr(GLMClient, "_chat_once", fail_once)
    settings = policy()
    observe(TrafficTap(tmp_path, settings))
    client = TapGLMClient(ENDPOINT, "GLM-5.3-Flash-EXL3")
    assert drain(tmp_path, policy=settings, client=client)["status"] == "error"
    assert len(attempts) == 1
    with database(tmp_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == 1
    with pytest.raises(ValueError, match="single-attempt transport"):
        drain(tmp_path, policy=settings, client=GLMClient(ENDPOINT, "GLM-5.3-Flash-EXL3"))


def test_failed_event_isolated_and_explicit_retry_opens_after_repair(tmp_path):
    from glm_rsi_dream.predream.tap_store import inspect_spool, retry_failed
    settings = policy(principal_events_per_minute=10)
    tap = TrafficTap(tmp_path, settings)
    observe(tap)
    observe(tap, "The fictional workshop holds amber screws.", request_id="two")

    class FirstBrokenClient(FixtureClient):
        def chat(self, messages, **kwargs):
            if messages[-1]["content"] == "The fictional workshop has cobalt washers.":
                self.calls += 1
                return "password unsafe-generated-output"
            return super().chat(messages, **kwargs)

    client = FirstBrokenClient()
    result = drain(tmp_path, policy=settings, client=client)
    assert result == {"status": "error", "processed": 1, "queued": 1, "failed": 1}
    assert inspect_spool(tmp_path)["states"] == {"done": 1, "failed": 1}
    assert retry_failed(tmp_path)["retried"] == 1
    assert drain(tmp_path, policy=settings, client=FixtureClient())["queued"] == 1
    assert inspect_spool(tmp_path)["states"] == {"done": 2}


def test_event_call_limit_is_observable_and_reopens_with_larger_budget(tmp_path):
    from glm_rsi_dream.predream.tap_store import retry_failed
    settings = policy(calls_per_event=1)
    observe(TrafficTap(tmp_path, settings))
    assert drain(tmp_path, policy=settings, client=FixtureClient())["status"] == "event_call_limit"
    assert retry_failed(tmp_path)["retried"] == 1
    assert drain(tmp_path, policy=replace(settings, calls_per_event=3), client=FixtureClient())["queued"] == 1
