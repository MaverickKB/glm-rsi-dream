# SPDX-License-Identifier: Apache-2.0
"""Explicit observer callbacks used by the hash-pinned startup source overlay."""

import hashlib
import json
import os
import sys
import threading
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, overload

from .capture_tap import (
    CONFIG_SHA as CONFIG_SHA,
)
from .capture_tap import (
    CaptureTap,
    TapConfig,
    packet_bytes,
    validate_metadata,
)
from .capture_tap import (
    log_notice as log_notice,
)
from .capture_tap import requested as _requested

KEY = "tensorfold_dream.capture_only"
WIRE_MAGIC = 1_964_111


def file_sha256(path: Any) -> Any:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1_048_576):
            digest.update(block)
    return digest.hexdigest()


# Attached serving engines share this process-wide breaker. Standalone taps
# have their own frame for offline use, without installing a serving overlay.
PROCESS_CAPTURE = {"failures": 0, "disabled": False}
REQUEST_CAPTURE = threading.local()
FAILURE_LIMIT = 3


def _release(tap: Any, sid: Any) -> Any:
    capture = tap.active.pop(sid, None)
    if capture is not None:
        tap.reserved -= capture["reserve"]
        if hasattr(tap, "queued_names"):
            tap.queued_names.discard(capture["name"])


def _raise_capture(exc: Any, owner: Any=None) -> Any:
    raise exc



def _failure_notice(tap: Any, message: Any) -> Any:
    if tap is None:
        log_notice("[dream-capture] " + message)
        return
    stats = getattr(tap, "stats", None)
    if stats is not None:
        stats["skipped"] += 1
    rank = getattr(tap, "rank", "unknown")
    log_notice(f"[dream-capture rank={rank}] skipped: {message}")


def capture_call(operation: Any, *args: Any, tap: Any=None, sid: Any=None, default: Any=None,
                 startup: Any=False, force: Any=False, on_error: Any=None, reason: Any=None, reservation: Any=None, **kwargs: Any) -> Any:
    """The sole serving/capture exception boundary. BaseException propagates."""
    state = PROCESS_CAPTURE
    try:
        if tap is not None:
            state = getattr(tap, "containment", PROCESS_CAPTURE)
        if not force and state["disabled"]:
            if tap is not None and sid is not None:
                capture_call(_release, tap, sid, force=True, on_error=lambda error: None)
            if reservation is not None:
                capture_call(tap.drop_pending, reservation, force=True, on_error=lambda error: None)
            return default
        return operation(*args, **kwargs)
    except Exception as exc:
        if on_error is not None:
            return capture_call(on_error, exc, default=default, force=True)
        state["failures"] += 1
        if startup or state["failures"] >= FAILURE_LIMIT:
            state["disabled"] = True
        scope = getattr(REQUEST_CAPTURE, "scope", None)
        report_notice = True
        if scope is not None and (tap is None or tap is scope.get("tap")):
            scope["failed"] = True
            report_notice = not scope["notified"]
            scope["notified"] = True
        # Cleanup is dictionary-only. Filesystem cleanup belongs to the writer.
        if tap is not None and sid is not None:
            capture_call(_release, tap, sid, force=True, on_error=lambda error: None)
        if reservation is not None:
            capture_call(tap.drop_pending, reservation, force=True, on_error=lambda error: None)
        if startup and args:
            engine = args[-1]
            if hasattr(engine, "w"):
                capture_call(engine.w.meta.pop, KEY, None, force=True, on_error=lambda error: None)
        if not report_notice:
            return default
        message = reason or ("capture observer: " + type(exc).__name__)
        if state["disabled"]:
            message += "; capture disabled for process"
        capture_call(_failure_notice, tap, message, force=True, on_error=lambda error: None)
        return default


@overload
def contained(operation: Callable[..., Any], *, startup: bool = False) -> Callable[..., Any]: ...


@overload
def contained(operation: None = None, *, startup: bool = False) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...


def contained(operation: Any = None, *, startup: bool = False) -> Any:
    def decorate(callback: Any) -> Any:
        @wraps(callback)
        def call(*args: Any, **kwargs: Any) -> Any:
            tap = None
            sid = None
            # Resolve capture ownership within the same boundary as the callback.
            def invoke() -> Any:
                nonlocal tap, sid
                owner = args[0] if args else None
                weights = getattr(owner, "w", owner)
                meta = getattr(weights, "meta", {})
                tap = meta.get(KEY)
                state = getattr(tap, "containment", PROCESS_CAPTURE)
                if state["disabled"] and callback.__name__ in ("attach", "capture_request"):
                    return None
                if state["disabled"] and callback.__name__ == "startup_code":
                    return [1, 0, 0, 0]
                if callback.__name__ == "admitted":
                    sid = args[1]["sid"]
                elif callback.__name__ == "finished":
                    sid = args[1]
                return callback(*args, **kwargs)

            def fail(exc: Any) -> Any:
                return capture_call(_raise_capture, exc, *args[:1], tap=tap, sid=sid, startup=startup)

            result = capture_call(invoke, on_error=fail, force=True)
            if result is None and callback.__name__ == "startup_code":
                return [1, 0, 0, 0]
            return result
        return call
    if operation is None:
        return decorate
    return decorate(operation)


@contained
def startup_code() -> Any:

    config = TapConfig.from_env()
    if config is None:
        return [1, 0, 0, 0]
    if config.max_bytes >= 2 ** 31:
        raise ValueError("capture cap must fit the native int32 settings protocol")
    return [1, 1, config.max_bytes, config.copy_rows]


@contained(startup=True)
def attach(engine: Any, model_dir: Any) -> Any:
    config = TapConfig.from_env()
    if config is None:
        return
    w = engine.w
    if w.world != 2 or engine.parallel != 4 or w.cfg.streams != 4:
        raise ValueError("capture startup requires the recorded TP=2, parallel=4, HC=4 layout")
    if "tensorfold_dream.resident" in w.meta or "tensorfold_dream.resident" in sys.modules:
        raise ValueError("capture-only startup cannot contain a resident binding")
    if os.environ.get("LW_ADAPTER") or os.environ.get("LW_CAPTURE_DIR"):
        raise ValueError("capture-only startup cannot use the legacy LW bootstrap")
    model_dir = Path(model_dir)
    config_sha = file_sha256(model_dir / "config.json")
    if config_sha != CONFIG_SHA:
        raise ValueError("capture startup model config differs from the pinned model configuration")
    tokenizer_file = model_dir / "tokenizer.json"
    from glm_rsi_dream.manifest import MANIFEST
    pins = MANIFEST["engine"]
    if os.environ.get("TF_DREAM_CAPTURE_ENGINE_SHA256") != pins["snapshot_sha256"]:
        raise ValueError("capture startup engine snapshot identity mismatch")
    identity = {"config_sha256": config_sha,
                "tokenizer_sha256": file_sha256(tokenizer_file),
                "model_snapshot": model_dir.name,
                "engine_snapshot_sha256": os.environ["TF_DREAM_CAPTURE_ENGINE_SHA256"]}
    tap = CaptureTap(config, w.rank, w.cfg.hidden, engine.tp.shared, w.cfg.top_k,
                     w.norm, w.cfg.eps, identity)
    from tensorfold.families.glm5_next.cuda.forward import glue

    tap.warm_combiner(glue.combine)
    tap.containment = PROCESS_CAPTURE
    w.meta[KEY] = tap
    log_notice(f"[dream-capture rank={w.rank}] observer ready; cap={config.max_bytes}; "
               f"metadata=glm-terminal-v1; warmup_s={tap.stats['warmup_seconds']:.3f}")



def requested(metadata: Any) -> Any:
    return capture_call(_requested, metadata, default=False)


def http_metadata(metadata: Any) -> Any:
    def select() -> Any:
        return {"dream_capture": metadata["dream_capture"]}
    return capture_call(select)


def _set_metadata(engine: Any, metadata: Any) -> Any:
    engine.request.metadata = metadata
    return True


def _metadata_scope(engine: Any, metadata: Any) -> Any:
    request = engine.request
    previous = getattr(request, "metadata", None)
    weights = getattr(engine, "w", None)
    tap = None if weights is None else weights.meta.get(KEY)
    scope = {"failed": False, "notified": False, "pending": [], "metadata": metadata, "engine": engine, "tap": tap}
    REQUEST_CAPTURE.scope = scope
    capture_call(_set_metadata, engine, metadata, tap=tap)
    return previous, tap, scope


@contextmanager
def request_metadata(engine: Any, metadata: Any) -> Any:
    """Always yield to native serving, even if capture setup or restore fails."""
    outer = getattr(REQUEST_CAPTURE, "scope", None)
    frame = capture_call(_metadata_scope, engine, metadata)
    try:
        yield
    finally:
        if frame is not None:
            previous, tap, scope = frame
            capture_call(_set_metadata, engine, previous, tap=tap, force=True)
            REQUEST_CAPTURE.scope = outer
            for pending_tap, packet in scope["pending"]:
                if scope["failed"]:
                    capture_call(pending_tap.drop_pending, packet, force=True)
                else:
                    capture_call(pending_tap.enqueue, packet, tap=pending_tap, reservation=packet)
        else:
            REQUEST_CAPTURE.scope = outer


@contained
def capture_request(engine: Any, prompt: Any, metadata: Any, vision: Any) -> Any:
    tap = engine.w.meta.get(KEY)
    if tap is None:
        return None
    scope = getattr(REQUEST_CAPTURE, "scope", None)
    if scope is not None and scope["engine"] is engine:
        if scope["failed"]:
            return None
        if metadata is None:
            metadata = scope["metadata"]
    # Metadata comes only from the explicit argument or active capture scope.
    # A failed engine-thread-local restore can therefore never leak a tag.
    if vision is not None:
        if requested(metadata):
            capture_call(tap.notice, "skipped", "image request cannot form a plain training lesson", tap=tap)
        return None
    if not requested(metadata):
        return None
    if packet_bytes(len(prompt), tap.width, tap.inputs) + 131_072 > tap.config.max_bytes:
        capture_call(tap.notice, "skipped", "request exceeds rank byte cap", tap=tap)
        return None
    return validate_metadata(metadata, prompt, tap.identity["config_sha256"])


def _wire_extension(spec: Any) -> Any:
    if spec is None:
        return [WIRE_MAGIC, 0]
    data = json.dumps(spec, separators=(",", ":")).encode()
    if len(data) > 65_536:
        raise ValueError("capture wire metadata exceeds bound")
    return [WIRE_MAGIC, len(data), *data]


def wire_extension(spec: Any) -> Any:
    return capture_call(_wire_extension, spec, default=[WIRE_MAGIC, 0])


def parse_wire(payload: Any, offset: Any, *, raise_decode_errors: Any=False) -> Any:
    if offset + 2 > len(payload) or payload[offset] != WIRE_MAGIC:
        raise ValueError("capture protocol marker missing; both ranks need the same overlay")
    length = payload[offset + 1]
    if not 0 <= length <= 65_536 or offset + 2 + length != len(payload):
        raise ValueError("capture protocol length mismatch")
    if not length:
        return None
    try:
        return json.loads(bytes(payload[offset + 2:]).decode())
    except Exception:
        if raise_decode_errors:
            raise
        return None


def parse_admit_capture(record: Any, payload: Any, offset: Any) -> Any:
    def failed_parse(exc: Any) -> Any:
        record["capture_error"] = type(exc).__name__
        return None

    def parse_and_insert() -> Any:
        record["capture"] = capture_call(parse_wire, payload, offset, raise_decode_errors=True,
                                         on_error=failed_parse, force=True)
        return record

    capture_call(parse_and_insert, on_error=failed_parse, force=True)
    return record


def _discard(tap: Any, sid: Any, reason: Any="incomplete native forward") -> Any:
    capture = tap.active.get(sid)
    reserved = tap.reserved
    capture_call(tap.discard, sid, reason, tap=tap, sid=sid, force=True)
    # A callback may have popped the object before raising.
    if capture is not None and sid not in tap.active:
        tap.reserved = min(tap.reserved, reserved - capture["reserve"])


@contained
def admitted(decoder: Any, record: Any, stream: Any) -> Any:
    tap = decoder.w.meta.get(KEY)
    framing_error = record.pop("capture_error", None)
    if tap is None:
        return
    if framing_error is not None:
        capture_call(_raise_capture, ValueError(framing_error), tap=tap, sid=record["sid"],
                     reason="ADMIT framing: " + framing_error)
        return
    capture_call(tap.begin, record["sid"], record["prompt"], record.get("capture"),
                 stream.cached, record["images"], tap=tap, sid=record["sid"])


@contained
def select_pieces(decoder: Any, group: Any) -> Any:
    tap = decoder.w.meta.get(KEY)
    pieces = []
    lo = 0
    for lane, stop in group:
        position = lane.st.pos
        count = stop - position
        if tap is not None:
            try:
                if lane.sid in tap.active and count > 0:
                    pieces.append((lane.sid, lo, position, count))
            except Exception as exc:
                capture_call(_raise_capture, exc, tap=tap, sid=lane.sid,
                             reason="piece observer: " + type(exc).__name__)
        lo += count
    decoder.e.pbuf._capture_pieces = pieces


@contained
def terminal(w: Any, b: Any, rows: Any) -> Any:
    tap = w.meta.get(KEY)
    pieces = getattr(b, "_capture_pieces", ())
    if tap is None or not b.prefill or not pieces:
        return
    from tensorfold.families.glm5_next.cuda.forward import glue

    for piece in pieces:
        capture_call(tap.terminal, b, [piece], glue.combine, tap=tap, sid=piece[0])


@contained
def completed(w: Any, b: Any, rows: Any) -> Any:
    tap = w.meta.get(KEY)
    pieces = getattr(b, "_capture_pieces", ())
    if tap is None or not b.prefill or not pieces:
        return
    for piece in pieces:
        capture_call(tap.completed, b, [piece], tap=tap, sid=piece[0])


@contained
def finished(decoder: Any, sid: Any) -> Any:
    tap = decoder.w.meta.get(KEY)
    if tap is not None:
        _discard(tap, sid)


@contained
def failed(w: Any, b: Any, reason: Any) -> Any:
    tap = w.meta.get(KEY)
    if tap is None:
        return
    for sid, lo, position, count in getattr(b, "_capture_pieces", ()):
        _discard(tap, sid, reason)
