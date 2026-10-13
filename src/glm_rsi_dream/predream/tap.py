"""Opt-in completed-request observers and a separate bounded intake collector."""

import fcntl
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .extract import GLMClient, discover_candidates
from .novelty import judge_novelty
from .queue import append_record, candidate_id, latest_records, validate_record
from .tap_policy import TapBudgetError, TapPolicy, privacy_reason
from .tap_store import (
    bind_context,
    capture,
    database,
    expire,
    opaque,
    private_directory,
    reserve_call,
)
from .timing import now_et


class TrafficTap:
    """Call from a trusted completion callback; consent never comes from text."""

    def __init__(self, root: Any, policy: Any = None, clock: Any = time.time, policy_path: Any = None) -> None:
        self.root = Path(root)
        self.policy = policy or TapPolicy()
        self.clock = clock
        self.policy_path = policy_path

    def observe(self, request: Any, *, principal: Any, request_id: Any, opt_in: Any = False,
                mode: Any, endpoint: Any, successful: Any = True, internal: Any = False) -> Any:
        if self.policy_path is not None:
            try:
                self.policy = TapPolicy.load(self.policy_path)
            except (OSError, ValueError):
                return {"status": "unavailable"}
        policy = self.policy
        if not policy.enabled:
            return {"status": "disabled"}
        if opt_in is not True or principal not in policy.principals:
            return {"status": "no_consent"}
        if mode != policy.mode or successful is not True or internal is not False:
            return {"status": "ineligible"}
        try:
            address = urlsplit(endpoint)
            if address.scheme not in {"http", "https"} or address.port != 8888 or not address.hostname:
                return {"status": "endpoint"}
            if endpoint.rstrip("/") != policy.endpoint.rstrip("/"):
                return {"status": "endpoint"}
            if address.username is not None or address.password is not None:
                return {"status": "endpoint"}
            if not isinstance(request_id, str) or not 0 < len(request_id) <= 256:
                return {"status": "request_id"}
            messages = request.get("messages")
            if not isinstance(messages, list) or not messages:
                return {"status": "unsupported"}
            # Only the current user turn is eligible; history never becomes evidence.
            message = messages[-1]
            if not isinstance(message, dict) or message.get("role") != "user":
                return {"status": "unsupported"}
            text = message.get("content")
            if not isinstance(text, str):
                return {"status": "unsupported"}
            reason = privacy_reason(text, policy)
            if reason:
                return {"status": reason}
            return capture(self.root, text, principal, request_id, policy, self.clock())
        except (AttributeError, TypeError, ValueError, OSError, sqlite3.Error):
            # Callback failures cannot break generation or reveal request material.
            return {"status": "unavailable"}

    def spark_completed(self, request: Any, **context: Any) -> Any:
        return self.observe(request, mode="spark", **context)

    def harness_completed(self, request: Any, **context: Any) -> Any:
        return self.observe(request, mode="harness", **context)


class TapGLMClient(GLMClient):
    """One HTTP attempt per budget reservation, including transport failures."""

    def chat(self, messages: Any, *, temperature: Any = None) -> Any:
        return self._chat_once(messages, temperature=temperature)


class BudgetClient:
    def __init__(self, client: Any, root: Any, policy: Any, clock: Any) -> None:
        self.client = client
        self.root = root
        self.policy = policy
        self.clock = clock
        self.calls = 0

    def chat(self, messages: Any, **kwargs: Any) -> Any:
        if self.calls >= self.policy.calls_per_event:
            raise TapBudgetError("event_call_limit")
        reserve_call(self.root, self.policy, self.clock())
        self.calls += 1
        answer = self.client.chat(messages, **kwargs)
        if privacy_reason(answer, self.policy):
            raise ValueError("Tap model output failed privacy check")
        return answer


def build_batch(event_id: Any, text: Any, *, client: Any, policy: Any) -> Any:
    started = time.perf_counter()
    candidates = discover_candidates(text, client=client)
    extract_ms = (time.perf_counter() - started) * 1000
    if len(candidates) > policy.candidates_per_event:
        raise ValueError("Tap candidate budget exhausted")
    batch = []
    source_ref = "tap:" + event_id
    for candidate in candidates:
        for field in ("text", "prompt_evidence", "probe_question"):
            if privacy_reason(candidate[field], policy):
                raise ValueError("Tap candidate failed privacy check")
        started = time.perf_counter()
        judgment = judge_novelty(candidate, client)
        probe_ms = (time.perf_counter() - started) * 1000
        record = {"id": candidate_id(source_ref, candidate["text"]), "created_et": now_et(),
                  "source": {"kind": "conversation", "ref": source_ref},
                  "text": candidate["text"], "prompt_evidence": candidate["prompt_evidence"],
                  "glm_prior": {"answer": judgment["answer"], "agrees": judgment["comparison"]["agrees"]},
                  "novelty": judgment["comparison"]["novelty"], "status": "pending", "grade": None,
                  "timings_ms": {"extract": extract_ms / max(1, len(candidates)),
                                 "probe": probe_ms, "grade": 0, "absorb": 0}}
        batch.append(record)
    return batch


def drain(root: Any, *, policy: Any, client: Any, max_events: Any = 4, clock: Any = time.time) -> Any:
    """Persist a safe batch before queue publication so interrupted writes resume."""
    if not policy.enabled:
        return {"status": "disabled", "processed": 0, "queued": 0}
    if isinstance(client, GLMClient) and not isinstance(client, TapGLMClient):
        raise ValueError("Tap requires a single-attempt transport client")
    if getattr(client, "trace_path", None) is not None:
        raise ValueError("Tap collector clients must have raw tracing disabled")
    if type(max_events) is not int or not 1 <= max_events <= policy.max_events:
        raise ValueError("max_events must be within the configured spool bound")
    directory = private_directory(Path(root) / "tap")
    descriptor = os.open(directory / "collector.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "busy", "processed": 0, "queued": 0}
        return _drain_locked(root, policy, client, max_events, clock)


def _drain_locked(root: Any, policy: Any, client: Any, max_events: Any, clock: Any) -> Any:
    result: dict[str, Any] = {"status": "ok", "processed": 0, "queued": 0}
    queue = Path(root) / "queue/candidates.jsonl"
    for iteration in range(max_events):
        with database(root) as connection:
            bind_context(connection, policy)
            expire(connection, policy, clock())
            event = connection.execute("SELECT id, text, state, batch, principal FROM events "
                                       "WHERE state IN ('captured','ready') ORDER BY created, id LIMIT 1").fetchone()
            if event is not None:
                allowed = []
                for name in policy.principals:
                    allowed.append(opaque(connection, name))
                if event[4] not in allowed:
                    connection.execute("DELETE FROM events WHERE id=?", (event[0],))
                    event = None
                    continue
        if event is None:
            break
        event_id, text, state, payload, principal = event
        try:
            if state == "captured":
                if privacy_reason(text, policy):
                    raise ValueError("Tap source failed current privacy policy")
                wrapped = BudgetClient(client, root, policy, clock)
                batch = build_batch(event_id, text, client=wrapped, policy=policy)
                payload = json.dumps(batch, ensure_ascii=False, allow_nan=False)
                with database(root) as connection:
                    size = connection.execute("SELECT COALESCE(SUM(size),0) FROM events WHERE id != ?",
                                              (event_id,)).fetchone()[0]
                    if size + len(payload.encode()) > policy.max_spool_bytes:
                        raise ValueError("Tap prepared batch exceeds capacity")
                    connection.execute("UPDATE events SET text=NULL, state='ready', batch=?, size=? WHERE id=?",
                                       (payload, len(payload.encode()), event_id))
            batch = json.loads(payload)
            # Recheck actual evidence/output fields, not timestamps or opaque IDs.
            for record in batch:
                validate_record(record)
                for value in (record["text"], record["prompt_evidence"], record["glm_prior"]["answer"]):
                    if privacy_reason(value, policy):
                        raise ValueError("Tap prepared batch failed current privacy policy")
            with database(root) as connection:
                expire(connection, policy, clock())
                allowed = []
                for name in policy.principals:
                    allowed.append(opaque(connection, name))
                revoked = connection.execute("SELECT 1 FROM revoked WHERE principal=?", (principal,)).fetchone()
                if principal not in allowed or revoked:
                    connection.execute("DELETE FROM events WHERE id=?", (event_id,))
                    continue
                if connection.execute("SELECT 1 FROM events WHERE id=?", (event_id,)).fetchone() is None:
                    continue
                current = latest_records(queue)
                seen = set()
                pending = 0
                for record in current:
                    seen.add(record["id"])
                    if record["status"] == "pending":
                        pending += 1
                new = []
                for record in batch:
                    if record["id"] not in seen:
                        new.append(record)
                        seen.add(record["id"])
                if pending + len(new) > policy.pending_candidates:
                    result["status"] = "queue_capacity"
                    break
                if new:
                    private_directory(queue.parent)
                    if queue.is_symlink():
                        raise ValueError("Tap queue must not be a symlink")
                    fd = os.open(queue, os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
                    os.close(fd)
                    os.chmod(queue, 0o600)
                for record in new:
                    append_record(queue, record)
                    result["queued"] += 1
                connection.execute("UPDATE events SET state='done', text=NULL, batch=NULL, size=0 WHERE id=?",
                                   (event_id,))
                result["processed"] += 1
        except TapBudgetError as error:
            result["status"] = str(error)
            if str(error) == "model_rate_limited":
                break
            with database(root) as connection:
                connection.execute("UPDATE events SET state='failed' WHERE id=?", (event_id,))
            result["failed"] = result.get("failed", 0) + 1
        except (ValueError, TypeError, KeyError, RuntimeError, OSError, sqlite3.Error):
            result["status"] = "error"
            # Isolate this event so later traffic progresses. Keep safe input for
            # explicit retry after repair, with no raw exceptions/model responses.
            with database(root) as connection:
                connection.execute("UPDATE events SET state='failed' WHERE id=?", (event_id,))
            result["failed"] = result.get("failed", 0) + 1
    return result


def observe_completion(tap: Any, request: Any, response: Any, **context: Any) -> Any:
    """External harness helper returning the exact response object it received."""
    tap.observe(request, **context)
    return response
