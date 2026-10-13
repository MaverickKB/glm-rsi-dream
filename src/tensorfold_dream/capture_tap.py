# SPDX-License-Identifier: Apache-2.0
"""Bounded, local terminal export. Serving tensors are read only.

Only prompt prefill rows are captured. No transformer, head, factor or optimizer
is constructed here. The native bridge supplies the original routed combiner.
"""

import hashlib
import json
import os
import queue
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

import torch

from glm_rsi_dream.manifest import MANIFEST

TAG = "glm-terminal-v1"
CONFIG_SHA = MANIFEST["model_config_sha256"]
FILE_ALLOWANCE = 131_072
METADATA_LIMIT = 65_536


# One bounded daemon drain is initialized before any serving request.
LOG_QUEUE: queue.Queue[str] = queue.Queue(maxsize=128)


def log_notice(message: Any) -> Any:
    try:
        LOG_QUEUE.put_nowait(message)
    except Exception:
        # Capture notices are best effort, including when the sink is stalled.
        pass


def _log_loop() -> Any:
    while True:
        message = LOG_QUEUE.get()
        try:
            print(message, flush=True)
        except Exception:
            pass
        finally:
            LOG_QUEUE.task_done()


LOG_WRITER = threading.Thread(target=_log_loop, name="dream-capture-log-writer", daemon=True)
LOG_WRITER.start()


def packet_layout(tokens: Any, width: Any, inputs: Any) -> Any:
    return {
        "sact": ((tokens, inputs), torch.bfloat16),
        "sy": ((tokens, width), torch.float32),
        "routed": ((tokens, width), torch.float32),
        "shared_weight": ((tokens,), torch.float32),
        "x": ((tokens, 4, width), torch.bfloat16),
        "post": ((tokens, 4), torch.float32),
        "comb": ((tokens, 4, 4), torch.float32),
        "hidden": ((tokens, width), torch.bfloat16),
        "fnormed": ((tokens, width), torch.bfloat16),
        "rows": ((tokens,), torch.int64),
        "token_ids": ((tokens,), torch.int64),
        "answer_mask": ((tokens,), torch.bool),
        "valid": ((tokens,), torch.bool),
        "owner_valid": ((tokens,), torch.bool),
        "norm": ((width,), torch.bfloat16),
    }


def packet_bytes(tokens: Any, width: Any, inputs: Any) -> Any:
    total = 0
    for shape, dtype in packet_layout(tokens, width, inputs).values():
        elements = 1
        for dimension in shape:
            elements *= dimension
        total += elements * dtype.itemsize
    return total


def requested(metadata: Any) -> Any:
    if not isinstance(metadata, dict):
        return False
    capture = metadata.get("dream_capture")
    return isinstance(capture, dict) and capture.get("tag") == TAG


def prepare_metadata(metadata: Any, prompt: Any, tokenizer: Any, config_sha: Any) -> Any:
    """Offline caller: verify plain completion text and derive answer token mask."""
    if not requested(metadata):
        return None
    spec = metadata["dream_capture"]
    request_id = spec.get("request_id")
    if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", request_id):
        raise ValueError("capture request_id must be 1..96 safe characters")
    text = spec.get("input_text")
    answer = spec.get("answer_start_character")
    if not isinstance(text, str) or len(text.encode()) > METADATA_LIMIT:
        raise ValueError("capture input_text exceeds metadata bound")
    if type(answer) is not int or not 0 < answer < len(text) or text[answer - 1] != "\n":
        raise ValueError("capture answer must follow the prompt newline")
    if spec.get("config_sha256") != config_sha:
        raise ValueError("capture model config mismatch")
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if encoded.ids != list(prompt):
        raise ValueError("capture needs exact plain-completion prompt+gold tokens")
    mask = []
    for start, stop in encoded.offsets:
        mask.append(stop > answer and stop > start)
    if not any(mask[1:]):
        raise ValueError("capture has no aligned answer targets")
    prepared = {"tag": TAG, "request_id": request_id, "input_text": text,
                "input_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "answer_start_character": answer, "answer_mask": mask,
                "token_ids": list(prompt),
                "config_sha256": config_sha}
    if len(json.dumps(prepared, separators=(",", ":")).encode()) > METADATA_LIMIT:
        raise ValueError("capture metadata exceeds wire bound")
    return prepared


def validate_metadata(metadata: Any, prompt: Any, config_sha: Any) -> Any:
    """Serving validates the caller's pretokenized capture identity.

    The offline producer uses prepare_metadata and must independently
    reproduce token IDs and answer boundaries before admitting a lesson.
    Serving retains no second tokenizer and does no alternate forward.
    """
    if not requested(metadata):
        return None
    spec = metadata["dream_capture"]
    expected = {"tag", "request_id", "input_text", "input_text_sha256", "answer_start_character",
                "answer_mask", "token_ids", "config_sha256"}
    if set(spec) != expected:
        raise ValueError("capture metadata field mismatch")
    request_id = spec.get("request_id")
    if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", request_id):
        raise ValueError("capture request_id must be 1..96 safe characters")
    if spec.get("config_sha256") != config_sha or spec.get("token_ids") != list(prompt):
        raise ValueError("capture model or token identity mismatch")
    mask = spec.get("answer_mask")
    if not isinstance(mask, list) or len(mask) != len(prompt):
        raise ValueError("capture answer mask length mismatch")
    for value in mask:
        if type(value) is not bool:
            raise ValueError("capture answer mask must contain booleans")
    if not any(mask[1:]):
        raise ValueError("capture has no aligned answer targets")
    text = spec.get("input_text")
    answer = spec.get("answer_start_character")
    if not isinstance(text, str) or len(text.encode()) > METADATA_LIMIT:
        raise ValueError("capture input text exceeds bound")
    if type(answer) is not int or not 0 < answer < len(text) or text[answer - 1] != "\n":
        raise ValueError("capture answer boundary must follow the prompt newline")
    if hashlib.sha256(text.encode()).hexdigest() != spec.get("input_text_sha256"):
        raise ValueError("capture input text digest mismatch")
    if len(json.dumps(spec, separators=(",", ":")).encode()) > METADATA_LIMIT:
        raise ValueError("capture metadata exceeds wire bound")
    return dict(spec)


@dataclass(frozen=True)
class TapConfig:
    directory: Path
    max_bytes: int = 64 * 1_048_576
    copy_rows: int = 32

    @classmethod
    def from_env(cls, env: Any=None) -> Any:
        env = os.environ if env is None else env
        enabled = env.get("TF_DREAM_CAPTURE", "0")
        if enabled == "0":
            return None
        if enabled != "1":
            raise ValueError("TF_DREAM_CAPTURE must be 0 or 1")
        directory = Path(env["TF_DREAM_CAPTURE_DIR"])
        cap = int(env.get("TF_DREAM_CAPTURE_MAX_BYTES", "67108864"))
        rows = int(env.get("TF_DREAM_CAPTURE_COPY_ROWS", "32"))
        if not directory.is_absolute() or cap <= FILE_ALLOWANCE or not 1 <= rows <= 32:
            raise ValueError("capture requires absolute directory, positive cap and 1..32 copy rows")
        return cls(directory, cap, rows)


class LimitedWriter:
    """Torch's zip writer streams directly to disk within the reservation."""

    def __init__(self, handle: Any, limit: Any) -> None:
        self.handle = handle
        self.limit = limit
        self.largest_write_bytes = 0

    def write(self, data: Any) -> Any:
        if self.handle.tell() + len(data) > self.limit:
            raise ValueError("serialized packet exceeds reservation")
        self.largest_write_bytes = max(self.largest_write_bytes, len(data))
        return self.handle.write(data)

    def flush(self) -> Any:
        self.handle.flush()


class CaptureTap:
    def __init__(self, config: Any, rank: Any, width: Any, inputs: Any, top_k: Any, norm: Any, norm_eps: Any, identity: Any) -> None:
        if rank not in (0, 1) or width % 128 or top_k <= 0:
            raise ValueError("capture requires the TP=2 terminal layout")
        self.config, self.rank = config, rank
        self.width, self.inputs, self.top_k = width, inputs, top_k
        self.norm, self.norm_eps, self.identity = norm, norm_eps, identity
        self.active: dict[Any, dict[str, Any]] = {}
        self.containment = {"failures": 0, "disabled": False}
        self.write_queue: queue.Queue[tuple[dict[str, Any], threading.Event]] = queue.Queue(maxsize=2)
        self.writer = None
        self.queued_names: set[str] = set()
        self.published_names: set[str] = set()
        self.reserved = 0
        self.stats = {"completed": 0, "skipped": 0, "failed": 0, "seconds": 0.0,
                      "largest_serialization_write_bytes": 0, "warmup_seconds": 0.0, "dropped": 0}
        config.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.disk_bytes = 0
        for file in config.directory.rglob("*"):
            if file.is_file():
                self.disk_bytes += file.stat().st_size
                self.published_names.add(file.parent.name)
        # Thread creation belongs to initialization, never to a request.
        self.writer = threading.Thread(target=self._write_loop, name="dream-capture-writer", daemon=True)
        self.writer.start()

    def notice(self, status: Any, reason: Any) -> Any:
        from .capture_native import capture_call

        def report() -> Any:
            self.stats[status] += 1
            log_notice(f"[dream-capture rank={self.rank}] {status}: {reason}")

        capture_call(report, tap=self, force=True)

    def warm_combiner(self, combine: Any) -> Any:
        """Warm the routed-only variant at startup on private one-row zeros.

        A one-row grid shares the later capture's D/SLOTS/BLOCK specialization.
        This moves first-use compilation out of the shared request scheduler.
        """
        start_time = time.perf_counter()
        device = self.norm.device
        with torch.inference_mode(False), torch.no_grad():
            experts = torch.zeros((1, self.top_k, self.width), dtype=torch.float32, device=device)
            weights = torch.zeros((1, self.top_k), dtype=torch.float32, device=device)
            output = torch.empty((1, self.width), dtype=torch.float32, device=device)
            combine(experts, weights, output)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        del experts, weights, output
        self.stats["warmup_seconds"] = time.perf_counter() - start_time

    def begin(self, sid: Any, prompt: Any, spec: Any, cached: Any=0, images: Any=False) -> Any:
        if spec is None or self.containment["disabled"]:
            return
        if cached or images:
            self.notice("skipped", "cached prefix or image rows cannot form a complete plain lesson")
            return
        tokens = len(prompt)
        size = packet_bytes(tokens, self.width, self.inputs)
        reserve = size + FILE_ALLOWANCE
        if self.disk_bytes + self.reserved + reserve > self.config.max_bytes:
            self.notice("skipped", "rank byte cap")
            return
        name = f"{spec['request_id']}-rank{self.rank}"
        active_name = False
        for capture in self.active.values():
            if capture["name"] == name:
                active_name = True
        if name in self.published_names or name in self.queued_names or sid in self.active or active_name:
            self.notice("skipped", "duplicate capture identity")
            return
        try:
            packet = {}
            with torch.inference_mode(False), torch.no_grad():
                for field, (shape, dtype) in packet_layout(tokens, self.width, self.inputs).items():
                    packet[field] = torch.zeros(shape, dtype=dtype, device="cpu")
                packet["rows"].copy_(torch.arange(tokens))
                packet["token_ids"].copy_(torch.tensor(prompt, dtype=torch.int64))
                packet["answer_mask"].copy_(torch.tensor(spec["answer_mask"], dtype=torch.bool))
                packet["norm"].copy_(self.norm.detach().cpu())
                completed_rows = torch.zeros(tokens, dtype=torch.bool, device="cpu")
                # Reserve only for this admitted tagged request. Terminal capture
                # reuses these buffers instead of asking the serving allocator.
                copy_rows = min(tokens, self.config.copy_rows)
                device = self.norm.device
                experts = torch.zeros((copy_rows, self.top_k, self.width), dtype=torch.float32, device=device)
                weights = torch.zeros((copy_rows, self.top_k), dtype=torch.float32, device=device)
                routed = torch.zeros((copy_rows, self.width), dtype=torch.float32, device=device)
            self.active[sid] = {"packet": packet, "spec": spec, "reserve": reserve, "name": name,
                                "completed_rows": completed_rows,
                                "experts": experts, "weights": weights, "routed": routed}
            self.reserved += reserve
        except Exception as exc:
            self._fault(sid, exc, "allocation: " + type(exc).__name__)

    def discard(self, sid: Any, reason: Any="incomplete native forward") -> Any:
        capture = self.active.pop(sid, None)
        if capture is not None:
            self.reserved -= capture["reserve"]
            self.notice("skipped", reason)

    def terminal(self, native: Any, pieces: Any, combine: Any) -> Any:
        """After terminal moe_block, before hc_post overwrites owner x.

        pieces are (sid, native row offset, request token offset, row count).
        Selected expert rows use buffers reserved at tagged admission for the
        unchanged native combiner. Copies use the current execution stream.
        Blocking host copies wait for that stream, never the whole device.
        """
        start_time = time.perf_counter()
        split = native.split
        if split is not None and split.active:
            owner_lo, owner_hi = split.own()
        elif self.rank == 0:
            owner_lo, owner_hi = 0, native.rows
        else:
            owner_lo, owner_hi = 0, 0
        for sid, lo, position, count in pieces:
            capture = self.active.get(sid)
            if capture is None:
                continue
            packet = capture["packet"]
            try:
                if not 0 <= position < position + count <= len(packet["rows"]):
                    raise ValueError("row provenance exceeds request")
                for offset in range(0, count, self.config.copy_rows):
                    take = min(self.config.copy_rows, count - offset)
                    first, stop = lo + offset, lo + offset + take
                    target = slice(position + offset, position + offset + take)
                    if bool(packet["valid"][target].any()):
                        raise ValueError("duplicate terminal rows")
                    if not torch.equal(native.ids[first:stop].cpu(), packet["token_ids"][target]):
                        raise ValueError("native token row mismatch")
                    with torch.inference_mode(False), torch.no_grad():
                        packet["sact"][target].copy_(native.sact[first:stop])
                        packet["sy"][target].copy_(native.sy[first:stop])
                        experts = capture["experts"][:take]
                        weights = capture["weights"][:take]
                        routed = capture["routed"][:take]
                        experts.copy_(native.ey[first:stop, :self.top_k])
                        weights.copy_(native.wts[first:stop, :self.top_k])
                        combine(experts, weights, routed)
                        packet["routed"][target].copy_(routed)
                        packet["shared_weight"][target].copy_(native.wts[first:stop, self.top_k])
                        a, z = max(first, owner_lo), min(stop, owner_hi)
                        if a < z:
                            dest = slice(position + a - lo, position + z - lo)
                            packet["x"][dest].copy_(native.x[a:z].view(z - a, 4, self.width))
                            packet["post"][dest].copy_(native.post[a:z])
                            packet["comb"][dest].copy_(native.comb[a:z].view(z - a, 4, 4))
                            packet["owner_valid"][dest] = True
                        packet["valid"][target] = True
            except Exception as exc:
                self._fault(sid, exc, "terminal copy failed: " + type(exc).__name__)
        self.stats["seconds"] += time.perf_counter() - start_time

    def completed(self, native: Any, pieces: Any) -> Any:
        """After the native chunk and any requested heads succeed, before reuse."""
        start_time = time.perf_counter()
        for sid, lo, position, count in pieces:
            capture = self.active.get(sid)
            if capture is None:
                continue
            packet = capture["packet"]
            try:
                for offset in range(0, count, self.config.copy_rows):
                    take = min(self.config.copy_rows, count - offset)
                    first, stop = lo + offset, lo + offset + take
                    target = slice(position + offset, position + offset + take)
                    with torch.inference_mode(False), torch.no_grad():
                        packet["hidden"][target].copy_(native.hidden[first:stop])
                        packet["fnormed"][target].copy_(native.fnormed[first:stop])
                        capture["completed_rows"][target] = True
                if position + count == len(packet["rows"]):
                    if not bool(packet["valid"].all()) or not bool(capture["completed_rows"].all()):
                        raise ValueError("missing terminal rows")
                    from .capture_native import capture_call

                    capture_call(self.publish, sid, tap=self, sid=sid)
            except Exception as exc:
                self._fault(sid, exc, "completion failed: " + type(exc).__name__)
        self.stats["seconds"] += time.perf_counter() - start_time

    def _fault(self, sid: Any, exc: Any, reason: Any) -> Any:
        from .capture_native import _raise_capture, capture_call

        capture_call(_raise_capture, exc, tap=self, sid=sid, reason=reason)

    def publish(self, sid: Any) -> Any:
        """Detach after the native head, then wait for successful scope restore."""
        from .capture_native import REQUEST_CAPTURE, capture_call

        capture = self.active[sid]
        self.queued_names.add(capture["name"])
        scope = getattr(REQUEST_CAPTURE, "scope", None)
        if scope is not None:
            scope["pending"].append((self, capture))
            self.active.pop(sid)
            return
        self.active.pop(sid)
        capture_call(self.enqueue, capture, tap=self, reservation=capture)

    def drop_pending(self, capture: Any) -> Any:
        if capture.get("queued", False):
            capture["cancelled"] = True
            return
        self.queued_names.discard(capture["name"])
        self.reserved -= capture["reserve"]

    def enqueue(self, capture: Any) -> Any:
        """Drop on full. The optional fast-writer wait has a fixed 20ms ceiling."""
        done = threading.Event()
        capture["queued"] = True
        try:
            self.write_queue.put_nowait((capture, done))
        except queue.Full:
            capture["queued"] = False
            self.drop_pending(capture)
            self.stats["dropped"] += 1
            from .capture_native import capture_call

            capture_call(self.notice, "skipped", "writer queue full; capture dropped", tap=self)
            return
        # Preserve immediate receipts when local storage is fast. A stalled
        # serializer/fsync can never extend this fixed request-path budget.
        done.wait(0.02)

    def _write_loop(self) -> Any:
        from .capture_native import capture_call

        while True:
            capture, done = self.write_queue.get()
            capture_call(self._write_capture, capture, tap=self)
            self.queued_names.discard(capture["name"])
            self.reserved -= capture["reserve"]
            done.set()
            self.write_queue.task_done()

    def _write_capture(self, capture: Any) -> Any:
        if capture.get("cancelled", False):
            return
        packet = capture["packet"]
        destination = self.config.directory / capture["name"]
        temporary = self.config.directory / ("." + capture["name"] + ".partial")
        temporary.mkdir(mode=0o700)
        try:
            packet_file = temporary / "packet.pt"
            with packet_file.open("xb") as handle:
                writer = LimitedWriter(handle, capture["reserve"] - METADATA_LIMIT)
                torch.save(packet, cast(BinaryIO, writer))
                self.stats["largest_serialization_write_bytes"] = max(
                    self.stats["largest_serialization_write_bytes"], writer.largest_write_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            digest = hashlib.sha256()
            with packet_file.open("rb") as handle:
                while block := handle.read(1_048_576):
                    digest.update(block)
            receipt = dict(capture["spec"])
            receipt.pop("answer_mask")
            receipt.pop("input_text")
            receipt.pop("token_ids", None)
            receipt.update(self.identity)
            receipt.update(rank=self.rank, tp_world=2, tokens=len(packet["rows"]),
                           norm_eps=self.norm_eps, completed_native_forward=True,
                           base_adapter_attached=False, packet_sha256=digest.hexdigest(),
                           packet_payload_bytes=packet_bytes(len(packet["rows"]), self.width, self.inputs))
            data = (json.dumps(receipt, sort_keys=True) + "\n").encode()
            if len(data) > METADATA_LIMIT:
                raise ValueError("receipt exceeds metadata bound")
            with (temporary / "receipt.json").open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            self._sync_directory(temporary)
            if destination.exists():
                raise ValueError("capture already exists")
            actual_bytes = packet_file.stat().st_size + len(data)
            if capture.get("cancelled", False):
                shutil.rmtree(temporary, ignore_errors=True)
                return
            os.rename(temporary, destination)
            self.disk_bytes += actual_bytes
            self.published_names.add(capture["name"])
            self._sync_directory(self.config.directory)
            self.stats["completed"] += 1
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    @staticmethod
    def _sync_directory(path: Any) -> Any:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
