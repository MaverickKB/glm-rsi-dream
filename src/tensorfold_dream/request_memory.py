# SPDX-License-Identifier: Apache-2.0
"""Reserve effective prompts and agree memory installation before any forward."""

KEY = "tensorfold_dream.eval"


def gather_status(decoder, values):
    """A failed rank exchange makes normal serving unavailable until recovery."""
    try:
        return decoder.g._gather_ints(values)
    except Exception as error:
        decoder.broken = error
        raise


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


def complete_admission(decoder, sid):
    """Vote on geometry, restore locally, then vote before any model forward."""
    from .attention_native import install

    state = decoder.w.meta[KEY]
    record = state["active"][sid]
    lane = decoder.lanes[sid]
    ready = int(not record.get("admission_error"))
    geometry = [ready, record["effective_length"], lane.s.count, lane.st.capacity]
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
        votes = gather_status(decoder, [int(not record.get("admission_error"))])
        for vote in votes:
            if not vote[0]:
                agreed = False
    if not agreed:
        message = "Paired memory installation rejected this request"
        record["admission_error"] = record.get("admission_error") or message
        record.pop("memory_plan", None)
        # This slot belongs only to this request. Both ranks release it alike.
        lane.st.reset()
        decoder._finish(sid)
        if decoder.w.rank == 0:
            raise ValueError(message)
        return
    lane.st._eval_sid = sid
    if hasattr(lane.s, "_memory_record"):
        del lane.s._memory_record
