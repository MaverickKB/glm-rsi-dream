# SPDX-License-Identifier: Apache-2.0
"""Reserve effective prompts and agree memory installation before any forward."""

KEY = "tensorfold_dream.eval"


def gather_status(decoder, values, phase="geometry"):
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


def make_record(w, spec, prompt, count, failures):
    from .attention_native import prepare
    from .pocket_native import admitted as initialize

    record = {"spec": spec, "main_calls": 0, "failures_at_start": failures}
    initialize(w, record, prompt)
    length = len(prompt)
    if spec["arm"] == "main" and record["policy"].get("use_memory"):
        length = prepare(w, record, prompt, count)
    record["effective_length"] = length
    return record


def prepare_request(decoder, stream):
    """Head-only preflight. A rejected request has emitted no ADMIT operation."""
    from .eval_native import STATE, selection

    spec = (getattr(stream, "glm", None) or {}).get("eval")
    if spec is None:
        return None
    cached = getattr(stream, "_memory_record", None)
    if cached is not None:
        return cached
    state = decoder.w.meta[KEY]
    checked = selection({"dream_eval": spec}, state["pair_sha256"])
    record = make_record(decoder.w, checked, stream.prompt, stream.count,
                         STATE.containment["failures"])
    if record["effective_length"] + stream.count > decoder.g.limit:
        raise ValueError("Memory-backed request exceeds the engine context limit")
    stream._memory_record = record
    return record


def extent_need(decoder, stream):
    record = prepare_request(decoder, stream)
    length = len(stream.prompt)
    if record is not None:
        length = max(length, record["effective_length"])
    # Native decoding grows extents as needed. Reserve prefill and its window,
    # rather than requiring the entire requested reply in the initial extent.
    return decoder._need(length)


def admitted(decoder, admission):
    """Both ranks register locally; errors wait for the paired agreement."""
    from .eval_native import STATE, selection

    spec = admission.get("eval")
    if spec is None:
        return
    state = decoder.w.meta[KEY]
    sid = admission["sid"]
    lane = decoder.lanes[sid]
    record = {"spec": spec, "main_calls": 0,
              "failures_at_start": STATE.containment["failures"]}
    try:
        checked = selection({"dream_eval": spec}, state["pair_sha256"])
        record = getattr(lane.s, "_memory_record", None)
        if record is None:
            record = make_record(decoder.w, checked, admission["prompt"],
                                 admission["count"], STATE.containment["failures"])
        if sid in state["active"] or len(state["active"]) >= decoder.count:
            raise ValueError("duplicate or over-capacity memory admission")
    except Exception as error:
        record = {"spec": spec, "main_calls": 0,
                  "failures_at_start": STATE.containment["failures"],
                  "admission_error": str(error), "effective_length": 0}
    record["sid"] = sid
    record["reserved_capacity"] = lane.st.capacity
    state["active"][sid] = record


def _zero_tree(value):
    if value is None or isinstance(value, (int, float, str, bool)):
        return
    zero = getattr(value, "zero_", None)
    if callable(zero):
        zero()
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _zero_tree(item)


def release_slot(lane):
    """Drop request-owned KDA state and attention rows written before rejection."""
    reset = getattr(lane.st, "reset", None)
    if callable(reset):
        reset()
    for name in ("kc", "vc", "mtp_kc", "mtp_vc", "index"):
        _zero_tree(getattr(lane.st, name, None))


def _reject_admission(decoder, record, sid):
    message = "Paired memory installation rejected this request"
    record["admission_error"] = record.get("admission_error") or message
    record.pop("memory_plan", None)
    # Release before _finish, which drops the lane.
    release_slot(decoder.lanes[sid])
    decoder._finish(sid)


def complete_admission(decoder, sid):
    """Vote on geometry, restore locally, then vote before any model forward."""
    from .attention_native import install

    state = decoder.w.meta[KEY]
    record = state["active"][sid]
    lane = decoder.lanes[sid]
    ready = int(not record.get("admission_error"))
    geometry = [ready, record["effective_length"], lane.s.count, lane.st.capacity]
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


def _join_reject(decoder, words):
    try:
        decoder.g._gather_ints(list(words))
    except Exception as error:
        if decoder.broken is None:
            decoder.broken = error


def _release_open_admission(decoder):
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


def unblock_peer(decoder):
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
