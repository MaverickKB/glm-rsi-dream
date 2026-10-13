# SPDX-License-Identifier: Apache-2.0
"""Reserve effective prompts and agree memory installation before any forward."""
from typing import Any, cast

KEY = "tensorfold_dream.eval"
# Native rank exchange and ADMIT payloads travel as signed int32 values.
INT32_MAX = 2_147_483_647


def fits_wire(value: object) -> bool:
    """True when a geometry field can cross the native int32 gather."""
    return type(value) is int and 0 <= value <= INT32_MAX


def wire_or_zero(value: object) -> int:
    if type(value) is int and 0 <= value <= INT32_MAX:
        return value
    return 0


def gather_status(decoder: Any, values: list[int], phase: str = "geometry") -> Any:
    """A failed rank exchange makes normal serving unavailable until recovery."""
    decoder._admission_phase = phase
    try:
        result = decoder.g._gather_ints(values)
    except Exception as error:
        decoder._admission_phase = "aborted"
        decoder.broken = error
        raise
    if phase == "geometry":
        decoder._admission_phase = "geometry-done"
    else:
        decoder._admission_phase = "done"
    return result


def counted_tokens(count: object) -> int:
    """Native admission turns a nonpositive count into one generated token."""
    if type(count) is not int:
        raise ValueError("Memory-backed request exceeds the engine context limit")
    if count < 1:
        return 1
    return count


def make_record(w: Any, spec: dict[str, Any], prompt: Any, count: Any, failures: int) -> dict[str, Any]:
    from .attention_native import prepare
    from .pocket_native import admitted as initialize

    record: dict[str, Any] = {"spec": spec, "main_calls": 0, "failures_at_start": failures}
    initialize(w, record, prompt)
    length = len(prompt)
    if spec["arm"] == "main" and record["policy"].get("use_memory"):
        length = prepare(w, record, prompt, count)
    record["effective_length"] = length
    return record


def prepare_request(decoder: Any, stream: Any) -> dict[str, Any] | None:
    """Head-only preflight. A rejected request has emitted no ADMIT operation."""
    from .eval_native import STATE, selection

    spec = (getattr(stream, "glm", None) or {}).get("eval")
    if spec is None:
        return None
    cached = getattr(stream, "_memory_record", None)
    if cached is not None:
        return cast(dict[str, Any], cached)
    state = decoder.w.meta[KEY]
    checked = selection({"dream_eval": spec}, state["pair_sha256"])
    record = make_record(decoder.w, checked, stream.prompt, stream.count,
                         STATE.containment["failures"])
    length = record["effective_length"]
    # A zero or negative count is raised to 1 after this check. Compare the
    # length that will actually be generated, not the raw requested count.
    counted = counted_tokens(stream.count)
    if type(length) is not int or length < 0 or type(decoder.g.limit) is not int:
        raise ValueError("Memory-backed request exceeds the engine context limit")
    if length + counted > decoder.g.limit:
        raise ValueError("Memory-backed request exceeds the engine context limit")
    if length > INT32_MAX or counted > INT32_MAX:
        raise ValueError("request extent does not fit the native int32 protocol")
    stream._memory_record = record
    return record


def extent_need(decoder: Any, stream: Any) -> int:
    record = prepare_request(decoder, stream)
    length = len(stream.prompt)
    if record is not None:
        length = max(length, record["effective_length"])
    if type(length) is not int or length < 0:
        raise ValueError("request extent does not fit the native int32 protocol")
    # Native decoding grows extents as needed. Reserve prefill and its window,
    # rather than requiring the entire requested reply in the initial extent.
    need = decoder._need(length)
    rows = getattr(decoder, "rows_max", None)
    if type(rows) is int and length + rows > need:
        raise ValueError("Request extent too small for memory prefill")
    if type(need) is not int or need <= 0 or need > INT32_MAX:
        raise ValueError("request extent does not fit the native int32 protocol")
    return need


def admitted(decoder: Any, admission: dict[str, Any]) -> None:
    """Both ranks register locally; errors wait for the paired agreement."""
    from .eval_native import STATE, selection

    spec = admission.get("eval")
    if spec is None:
        return
    state: dict[str, Any] | None = None
    sid: Any = None
    try:
        state = decoder.w.meta[KEY]
        sid = admission["sid"]
        lane = decoder.lanes[sid]
        checked = selection({"dream_eval": spec}, state["pair_sha256"])
        cached = getattr(lane.s, "_memory_record", None)
        if cached is None:
            record = make_record(decoder.w, checked, admission["prompt"],
                                 admission["count"], STATE.containment["failures"])
        else:
            record = cached
        if sid in state["active"] or len(state["active"]) >= decoder.count:
            raise ValueError("duplicate or over-capacity memory admission")
        length = record.get("effective_length")
        capacity = lane.st.capacity
        if not fits_wire(length):
            raise ValueError("admission length is not a native int32")
        if not fits_wire(capacity):
            raise ValueError("admission extent is not a native int32")
        record["reserved_capacity"] = capacity
    except Exception as error:
        record = {"spec": spec, "main_calls": 0,
                  "failures_at_start": STATE.containment["failures"],
                  "admission_error": str(error), "effective_length": 0}
        try:
            if state is None:
                state = decoder.w.meta[KEY]
            if sid is None:
                sid = admission["sid"]
            record["reserved_capacity"] = decoder.lanes[sid].st.capacity
        except Exception:
            record["reserved_capacity"] = 0
        # A request-local failure must not escape into the handler that marks
        # both ranks broken. Without a stream id there is nowhere to record it.
        if state is None or sid is None:
            return
    record["sid"] = sid
    state["active"][sid] = record


def _zero_tree(value: Any) -> None:
    if value is None or isinstance(value, (int, float, str, bool)):
        return
    zero = getattr(value, "zero_", None)
    if callable(zero):
        zero()
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _zero_tree(item)


def release_slot(lane: Any) -> None:
    """Drop request-owned KDA state and attention rows written before rejection."""
    reset = getattr(lane.st, "reset", None)
    if callable(reset):
        reset()
    for name in ("kc", "vc", "mtp_kc", "mtp_vc", "index"):
        _zero_tree(getattr(lane.st, name, None))


def _reject_admission(decoder: Any, record: Any, sid: Any) -> None:
    message = "Paired memory installation rejected this request"
    record["admission_error"] = record.get("admission_error") or message
    record.pop("memory_plan", None)
    # Release before _finish, which drops the lane.
    release_slot(decoder.lanes[sid])
    decoder._finish(sid)


def complete_admission(decoder: Any, sid: Any) -> None:
    """Vote on geometry, restore locally, then vote before any model forward."""
    from .attention_native import install

    state = decoder.w.meta[KEY]
    record = state["active"][sid]
    lane = decoder.lanes[sid]
    ready = int(not record.get("admission_error"))
    length = record.get("effective_length")
    count = getattr(lane.s, "count", None)
    capacity = getattr(getattr(lane, "st", None), "capacity", None)
    # An overflowing field must not reach the int32 gather. That gather marks
    # the whole pair broken, which takes down every other request.
    if ready and not (fits_wire(length) and fits_wire(count) and fits_wire(capacity)):
        message = "admission geometry exceeds the native int32 protocol"
        record["admission_error"] = record.get("admission_error") or message
        ready = 0
    geometry = [ready, wire_or_zero(length), wire_or_zero(count), wire_or_zero(capacity)]
    agreed = False
    try:
        votes = gather_status(decoder, geometry)
        agreed = True
        for vote in votes:
            if not vote[0] or vote != votes[0]:
                agreed = False
        if agreed:
            try:
                install(decoder, record)
            except Exception as error:
                record["admission_error"] = str(error)
            votes = gather_status(decoder, [int(not record.get("admission_error"))], "confirm")
            agreed = True
            for vote in votes:
                if not vote[0]:
                    agreed = False
    except Exception:
        # A failed exchange still owns this slot. Clear it before the error leaves.
        _reject_admission(decoder, record, sid)
        raise
    if not agreed:
        decoder._admission_phase = "done"
        _reject_admission(decoder, record, sid)
        if decoder.w.rank == 0:
            raise ValueError("Paired memory installation rejected this request")
        return
    lane.st._eval_sid = sid
    if hasattr(lane.s, "_memory_record"):
        del lane.s._memory_record


def _join_reject(decoder: Any, words: Any) -> None:
    try:
        decoder.g._gather_ints(list(words))
    except Exception as error:
        if decoder.broken is None:
            decoder.broken = error


def _release_open_admission(decoder: Any) -> None:
    state = getattr(decoder.w, "meta", {}).get(KEY)
    if not state:
        return
    sid = getattr(decoder, "_eval_admit", None)
    if sid is None or sid not in state.get("active", {}):
        return
    if sid not in decoder.lanes:
        state["active"].pop(sid, None)
        return
    _reject_admission(decoder, state["active"][sid], sid)


def unblock_peer(decoder: Any) -> None:
    """Join the admission vote a peer may still be waiting on, then drop this slot."""
    phase = getattr(decoder, "_admission_phase", None)
    if phase in ("aborted", "done", "geometry", "confirm"):
        decoder._eval_admit = None
        return
    if phase == "geometry-done":
        _join_reject(decoder, [0])
    elif getattr(decoder, "_eval_admit", None) is None:
        return
    else:
        _join_reject(decoder, [0, 0, 0, 0])
    _release_open_admission(decoder)
    decoder._admission_phase = "aborted"
    decoder._eval_admit = None
