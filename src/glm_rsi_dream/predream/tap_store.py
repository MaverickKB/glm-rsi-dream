"""Private durable tap spool, with serialized admission and sliding budgets."""

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .tap_policy import TapBudgetError


def private_directory(path: Any) -> Any:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or path.stat().st_uid != os.getuid():
        raise ValueError("Tap storage must be owned by this user and not a symlink")
    os.chmod(path, 0o700)
    return path


@contextmanager
def database(root: Any) -> Any:
    directory = private_directory(Path(root) / "tap")
    path = directory / "spool.sqlite3"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    os.close(descriptor)
    os.chmod(path, 0o600)
    connection = sqlite3.connect(path, timeout=0.05)
    try:
        connection.execute("PRAGMA secure_delete=ON")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        connection.execute("CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, principal TEXT, "
                           "created REAL, text TEXT, state TEXT, batch TEXT, size INTEGER)")
        connection.execute("CREATE TABLE IF NOT EXISTS calls (created REAL)")
        connection.execute("CREATE TABLE IF NOT EXISTS revoked (principal TEXT PRIMARY KEY)")
        row = connection.execute("SELECT value FROM meta WHERE key='key'").fetchone()
        if row is None:
            connection.execute("INSERT INTO meta VALUES ('key', ?)", (secrets.token_hex(32),))
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def opaque(connection: Any, value: Any) -> Any:
    key = connection.execute("SELECT value FROM meta WHERE key='key'").fetchone()[0]
    return hmac.new(bytes.fromhex(key), value.encode(), hashlib.sha256).hexdigest()


def expire(connection: Any, policy: Any, now: Any) -> Any:
    connection.execute("DELETE FROM events WHERE created < ?", (now - policy.retention_seconds,))
    connection.execute("DELETE FROM calls WHERE created <= ?", (now - 60,))


def bind_context(connection: Any, policy: Any) -> Any:
    identity = json.dumps([policy.mode, policy.endpoint.rstrip("/")])
    row = connection.execute("SELECT value FROM meta WHERE key='context'").fetchone()
    if row is None:
        connection.execute("INSERT INTO meta VALUES ('context', ?)", (identity,))
    elif row[0] != identity:
        raise ValueError("Tap mode or endpoint changed; use a new private root")


def capture(root: Any, text: Any, principal: Any, request_id: Any, policy: Any, now: Any) -> Any:
    with database(root) as connection:
        bind_context(connection, policy)
        expire(connection, policy, now)
        actor = opaque(connection, principal)
        event_id = opaque(connection, json.dumps([principal, request_id]))
        if connection.execute("SELECT 1 FROM revoked WHERE principal=?", (actor,)).fetchone():
            return {"status": "revoked"}
        if connection.execute("SELECT 1 FROM events WHERE id=?", (event_id,)).fetchone():
            return {"status": "duplicate"}
        count, size = connection.execute("SELECT COUNT(*), COALESCE(SUM(size),0) FROM events").fetchone()
        if count >= policy.max_events or size + len(text.encode()) > policy.max_spool_bytes:
            return {"status": "capacity"}
        recent = connection.execute("SELECT COUNT(*) FROM events WHERE created > ?", (now - 60,)).fetchone()[0]
        actor_recent = connection.execute("SELECT COUNT(*) FROM events WHERE principal=? AND created > ?",
                                          (actor, now - 60)).fetchone()[0]
        if recent >= policy.events_per_minute or actor_recent >= policy.principal_events_per_minute:
            return {"status": "rate_limited"}
        connection.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?)",
                           (event_id, actor, now, text, "captured", None, len(text.encode())))
        return {"status": "captured"}


def reserve_call(root: Any, policy: Any, now: Any) -> Any:
    with database(root) as connection:
        bind_context(connection, policy)
        expire(connection, policy, now)
        count = connection.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
        if count >= policy.model_calls_per_minute:
            raise TapBudgetError("model_rate_limited")
        connection.execute("INSERT INTO calls VALUES (?)", (now,))


def revoke(root: Any, principal: Any) -> Any:
    with database(root) as connection:
        actor = opaque(connection, principal)
        connection.execute("INSERT OR IGNORE INTO revoked VALUES (?)", (actor,))
        count = connection.execute("SELECT COUNT(*) FROM events WHERE principal=?", (actor,)).fetchone()[0]
        connection.execute("DELETE FROM events WHERE principal=?", (actor,))
        return {"status": "revoked", "purged_events": count}


def inspect_spool(root: Any) -> Any:
    with database(root) as connection:
        counts = {}
        for state, count in connection.execute("SELECT state, COUNT(*) FROM events GROUP BY state"):
            counts[state] = count
        return {"status": "ok", "states": counts}


def retry_failed(root: Any) -> Any:
    with database(root) as connection:
        count = connection.execute("SELECT COUNT(*) FROM events WHERE state='failed'").fetchone()[0]
        connection.execute("UPDATE events SET state=CASE WHEN batch IS NULL THEN 'captured' ELSE 'ready' END "
                           "WHERE state='failed'")
        return {"status": "ok", "retried": count}
