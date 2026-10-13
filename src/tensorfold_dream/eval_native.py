# SPDX-License-Identifier: Apache-2.0
"""Tagged native application. Capture's capture_call is the Exception boundary."""

import hashlib
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

from tensorfold_dream import capture_native as capture

KEY = "tensorfold_dream.eval"
TAG = "glm-eval-v1"
MAGIC = capture.WIRE_MAGIC
STATE = SimpleNamespace(rank="unknown", containment={"failures": 0, "disabled": False})


def call(operation, *args, default=None, startup=False, **kwargs):
    return capture.capture_call(
        operation, *args, tap=STATE, default=default, startup=startup, **kwargs
    )


def requested(metadata):
    return isinstance(metadata, dict) and "dream_eval" in metadata


def http_metadata(metadata):
    selected = {}
    if isinstance(metadata, dict):
        for name in ("dream_capture", "dream_eval"):
            if name in metadata:
                selected[name] = metadata[name]
    return selected


def selection(metadata, pair_sha):
    if not requested(metadata):
        return None
    if "dream_capture" in metadata:
        raise ValueError("capture and evaluation tags are mutually exclusive")
    spec = metadata["dream_eval"]
    if not isinstance(spec, dict) or set(spec) != {
        "tag",
        "request_id",
        "arm",
        "pair_sha256",
    }:
        raise ValueError("exact evaluation selection fields required")
    if spec["tag"] != TAG or spec["arm"] not in ("none", "main"):
        raise ValueError("unknown arm or tag")
    if (
        not isinstance(spec["request_id"], str)
        or re.fullmatch(r"[A-Za-z0-9_-]{1,96}", spec["request_id"]) is None
    ):
        raise ValueError("unsafe request identity")
    if spec["pair_sha256"] != pair_sha:
        raise ValueError("pair identity mismatch")
    return dict(spec)


def checked_file(root, record):
    path = (root / record["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("candidate file must remain in manifest directory")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("candidate digest mismatch: " + record["path"])
    return path


def load_candidates(path, w, inputs):
    from .pocket_native import load

    return load(path, w)


def _attach(engine, model_dir):
    w = engine.w
    if w.world != 2 or engine.parallel != 4 or len(w.layers) != 45:
        raise ValueError(
            "native evaluation requires TP=2, parallel=4, 45-layer GLM"
        )
    if capture.file_sha256(Path(model_dir) / "config.json") != capture.CONFIG_SHA:
        raise ValueError("native model config mismatch")
    state = load_candidates(os.environ["TF_DREAM_EVAL_MANIFEST"], w, engine.tp.shared)
    state["receipts"].mkdir(parents=True, exist_ok=True)
    w.meta[KEY] = state
    return state


def attach(engine, model_dir):
    if os.environ.get("TF_DREAM_EVAL") != "1":
        return
    STATE.rank = engine.w.rank
    state = call(_attach, engine, model_dir, startup=True)
    # Same native rank exchange is reached even after local factor load failure.
    identity = "0" * 64 if state is None else state["pair_sha256"]
    mine = [int(state is not None)] + list(bytes.fromhex(identity))
    rows = engine._gather_ints(mine)
    if any(row != rows[0] for row in rows) or not rows[0][0]:
        engine.w.meta.pop(KEY, None)
        STATE.containment["disabled"] = True
        capture.log_notice(
            f"[dream-eval rank={engine.w.rank}] unavailable; paired native fallback"
        )
    else:
        capture.log_notice(f"[dream-eval rank={engine.w.rank}] ready pair={identity}")


def request(engine, metadata):
    if os.environ.get("TF_DREAM_EVAL") != "1":
        return None
    scope = getattr(capture.REQUEST_CAPTURE, "scope", None)
    if metadata is None and scope is not None and scope["engine"] is engine:
        metadata = scope["metadata"]
    state = engine.w.meta.get(KEY)
    if state is None:
        return None
    return call(selection, metadata, state["pair_sha256"])


def wire_extension(capture_spec, eval_spec):
    # Native untagged ADMIT stays byte-identical, including its absent tail.
    if capture_spec is None and eval_spec is None:
        return []
    raw = json.dumps(
        {"capture": capture_spec, "eval": eval_spec}, separators=(",", ":")
    ).encode()
    if len(raw) > 65_536:
        raise ValueError("request extension exceeds bound")
    return [MAGIC, len(raw), *raw]


def parse_admit(record, payload, offset):
    if offset == len(payload):
        return record

    def parse():
        envelope = capture.parse_wire(payload, offset, raise_decode_errors=True)
        if set(envelope) != {"capture", "eval"}:
            raise ValueError("invalid request envelope")
        record["capture"] = envelope["capture"]
        record["eval"] = envelope["eval"]
        return record

    return call(parse, default=record)


def admitted(decoder, record):
    from .request_memory import admitted as register
    register(decoder, record)


def spec_for(w, st):
    state = getattr(w, "meta", {}).get(KEY)
    if state is None:
        return None
    return state["active"].get(getattr(st, "_eval_sid", None))


def tagged(w, st):
    return spec_for(w, st) is not None


def select_prefill(decoder, group):
    pieces = []
    offset = 0
    for lane, stop in group:
        count = stop - lane.st.pos
        if count > 0:
            pieces.append((lane.st, offset, count))
            offset += count
    decoder.e.pbuf._eval_pieces = pieces


def select_decode(w, b, segments):
    pieces = []
    offset = 0
    for seg in segments:
        count = len(seg.tokens)
        pieces.append((seg.st, offset, count))
        offset += count
    b._eval_pieces = pieces


def has_main(w, b):
    for st, offset, count in getattr(b, "_eval_pieces", ()):
        record = spec_for(w, st)
        if record is not None and record["spec"]["arm"] == "main":
            return True
    return False


def _finished(decoder, sid):
    state = decoder.w.meta.get(KEY)
    if state is None:
        return
    record = state["active"].pop(sid, None)
    if record is None:
        return
    receipt = dict(record["spec"])
    receipt.update(
        tp_rank=decoder.w.rank,
        tp_world=2,
        main_calls=record["main_calls"],
        hook_failed=STATE.containment["failures"] != record["failures_at_start"],
        completed=not bool(record.get('admission_error')),
        admission_error=record.get('admission_error'),
        reserved_capacity=record.get('reserved_capacity'),
    )
    from .pocket_native import finished as finalize

    if not record.get("admission_error"):
        receipt.update(finalize(decoder.w, record))
    path = state["receipts"] / (
        receipt["request_id"] + "-rank" + str(decoder.w.rank) + ".json"
    )
    with path.open("x") as target:
        json.dump(receipt, target, sort_keys=True)
        target.flush()
        os.fsync(target.fileno())
    capture.log_notice(
        f"[dream-eval rank={decoder.w.rank}] completed {receipt['request_id']}"
    )


def finished(decoder, sid):
    return call(_finished, decoder, sid)
