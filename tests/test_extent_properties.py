"""Property checks for extent sizing and paired memory admission."""
import io
from types import SimpleNamespace

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from tensorfold_dream import request_memory
from tensorfold_dream.capture_tap import LimitedWriter

INT32_MAX = request_memory.INT32_MAX
KEY = request_memory.KEY


class TokenPrompt:
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count


def align_need(rows_max):
    def need(length):
        return ((length + rows_max + 2047) // 2048) * 2048

    return need


def decoder(rank=0, limit=1048576, rows_max=32, capacity=2048):
    stream = SimpleNamespace(prompt=TokenPrompt(3), count=8)
    slot = SimpleNamespace(capacity=capacity, reset=lambda: None)
    lane = SimpleNamespace(s=stream, st=slot)
    state = {"active": {}, "prefix": {"normal_ids": [1]},
             "snapshot": SimpleNamespace(ids=[0] * 96), "pair_sha256": "a" * 64}
    value = SimpleNamespace(
        w=SimpleNamespace(rank=rank, meta={KEY: state}),
        g=SimpleNamespace(limit=limit, _gather_ints=None),
        lanes={0: lane}, rows_max=rows_max, count=4, broken=None)
    value._need = align_need(rows_max)

    def finish(sid):
        value.lanes.pop(sid, None)
        state["active"].pop(sid, None)

    value._finish = finish
    return value


def valid_spec():
    return {"tag": "glm-eval-v1", "request_id": "cpu-extent",
            "arm": "main", "pair_sha256": "a" * 64}


@given(
    prompt_len=st.integers(min_value=0, max_value=2**32),
    rows_max=st.integers(min_value=0, max_value=4096),
    effective=st.one_of(
        st.none(),
        st.integers(min_value=-10, max_value=3_000_000),
        st.floats(allow_nan=False, allow_infinity=False, width=32),
        st.booleans(),
    ),
)
def test_extent_is_bounded_or_rejected(prompt_len, rows_max, effective):
    value = decoder(rows_max=rows_max)
    stream = value.lanes[0].s
    stream.prompt = TokenPrompt(prompt_len)
    if effective is not None:
        stream._memory_record = {"effective_length": effective}
        stream.glm = {"eval": {"arm": "main"}}
    try:
        need = request_memory.extent_need(value, stream)
    except (ValueError, TypeError):
        assert value.broken is None
        return
    assert type(need) is int
    assert 0 < need <= INT32_MAX
    assert need % 2048 == 0
    assert value.broken is None


@given(
    length=st.integers(min_value=0, max_value=20_000),
    count=st.integers(min_value=-8, max_value=40_000),
    limit=st.integers(min_value=1, max_value=30_000),
)
def test_prepare_accepts_only_counted_context(length, count, limit, monkeypatch):
    value = decoder(limit=limit)
    stream = value.lanes[0].s
    stream.prompt = TokenPrompt(4)
    stream.count = count
    stream.glm = {"eval": valid_spec()}
    monkeypatch.setattr(request_memory, "make_record",
                        lambda *args: {"effective_length": length, "policy": {}})
    counted = count if count >= 1 else 1
    exceeds = length + counted > limit or length > INT32_MAX or counted > INT32_MAX
    if exceeds:
        try:
            request_memory.prepare_request(value, stream)
        except ValueError:
            assert not hasattr(stream, "_memory_record")
            assert value.broken is None
            assert value.w.meta[KEY]["active"] == {}
            return
        raise AssertionError("over-limit request was accepted")
    record = request_memory.prepare_request(value, stream)
    assert record["effective_length"] == length
    assert stream._memory_record is record
    assert value.broken is None
    assert stream.prompt.count == 4


@given(count=st.one_of(
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.none(),
    st.text(max_size=4),
))
def test_non_integer_count_fails_closed(count, monkeypatch):
    value = decoder(limit=128)
    stream = value.lanes[0].s
    stream.count = count
    stream.glm = {"eval": valid_spec()}
    monkeypatch.setattr(request_memory, "make_record",
                        lambda *args: {"effective_length": 3, "policy": {}})
    try:
        request_memory.prepare_request(value, stream)
    except ValueError as error:
        assert "engine context limit" in str(error)
    else:
        raise AssertionError("non-integer count was accepted")
    assert not hasattr(stream, "_memory_record")
    assert value.broken is None


@given(
    length=st.one_of(
        st.integers(min_value=-5, max_value=2**33),
        st.none(),
        st.booleans(),
        st.floats(allow_nan=False, allow_infinity=False, width=32),
    ),
    count=st.one_of(
        st.integers(min_value=-5, max_value=2**33),
        st.none(),
        st.booleans(),
    ),
    capacity=st.one_of(
        st.integers(min_value=-5, max_value=2**33),
        st.none(),
    ),
    rank=st.sampled_from((0, 1)),
)
def test_gather_receives_only_int32_geometry(length, count, capacity, rank):
    value = decoder(rank=rank, capacity=2048)
    value.lanes[0].s.count = count
    value.lanes[0].st.capacity = capacity
    value.w.meta[KEY]["active"][0] = {
        "effective_length": length,
        "spec": {"arm": "none"},
        "policy": {},
    }
    seen = []

    def gather(words):
        assert isinstance(words, list)
        for word in words:
            assert type(word) is int
            assert 0 <= word <= INT32_MAX
            torch.tensor([word], dtype=torch.int32)
        seen.append(list(words))
        return [[0, 0, 0, 0], [1, 0, 0, 0]]

    value.g._gather_ints = gather
    if rank == 0:
        try:
            request_memory.complete_admission(value, 0)
        except ValueError as error:
            assert "rejected this request" in str(error)
        else:
            raise AssertionError("rank 0 kept a disagreed admission")
    else:
        request_memory.complete_admission(value, 0)
    assert value.broken is None
    assert seen
    assert 0 not in value.lanes
    assert 0 not in value.w.meta[KEY]["active"]


def test_admitted_keeps_a_fitting_request(monkeypatch):
    value = decoder(capacity=4096)
    spec = valid_spec()
    monkeypatch.setattr(request_memory, "make_record",
                        lambda *args: {"effective_length": 96, "policy": {}, "spec": spec})
    request_memory.admitted(value, {"eval": spec, "sid": 0, "prompt": [1, 2, 3], "count": 4})
    record = value.w.meta[KEY]["active"][0]
    assert record["reserved_capacity"] == 4096
    assert "admission_error" not in record
    assert value.broken is None


def test_admitted_records_over_capacity_and_wide_lengths(monkeypatch):
    value = decoder()
    value.count = 0
    spec = valid_spec()
    monkeypatch.setattr(request_memory, "make_record",
                        lambda *args: {"effective_length": 2**40, "policy": {}, "spec": spec})
    request_memory.admitted(value, {"eval": spec, "sid": 0, "prompt": [1], "count": 1})
    record = value.w.meta[KEY]["active"][0]
    assert "admission_error" in record
    assert value.broken is None


def test_prepare_rejects_unusable_lengths(monkeypatch):
    value = decoder(limit=2**40)
    stream = value.lanes[0].s
    stream.glm = {"eval": valid_spec()}
    stream.count = 1
    monkeypatch.setattr(request_memory, "make_record",
                        lambda *args: {"effective_length": -1, "policy": {}})
    with pytest.raises(ValueError, match="engine context limit"):
        request_memory.prepare_request(value, stream)
    monkeypatch.setattr(request_memory, "make_record",
                        lambda *args: {"effective_length": 2**32, "policy": {}})
    with pytest.raises(ValueError, match="int32"):
        request_memory.prepare_request(value, stream)
    assert value.broken is None


def test_short_extent_is_rejected():
    value = decoder(rows_max=32)
    value._need = lambda length: length
    stream = value.lanes[0].s
    with pytest.raises(ValueError, match="too small"):
        request_memory.extent_need(value, stream)
    assert value.broken is None


def test_admitted_uses_a_cached_record_and_can_return_quietly(monkeypatch):
    value = decoder()
    spec = valid_spec()
    cached = {"effective_length": 12, "policy": {}, "spec": spec}
    value.lanes[0].s._memory_record = cached
    request_memory.admitted(value, {"eval": spec, "sid": 0, "prompt": [1], "count": 1})
    assert value.w.meta[KEY]["active"][0] is cached
    request_memory.admitted(value, {"prompt": [1]})
    assert 3 not in value.w.meta[KEY]["active"]
    value.w.meta.pop(KEY)
    request_memory.admitted(value, {"eval": spec})
    assert value.broken is None


def test_agreed_admission_clears_the_preflight_cache(monkeypatch):
    from tensorfold_dream import attention_native
    value = decoder()
    value.lanes[0].s._memory_record = {"cached": True}
    value.w.meta[KEY]["active"][0] = {"effective_length": 8, "spec": {"arm": "none"}, "policy": {}}
    value.g._gather_ints = lambda words: [list(words), list(words)]
    monkeypatch.setattr(attention_native, "install", lambda decoder, record: None)
    request_memory.complete_admission(value, 0)
    assert not hasattr(value.lanes[0].s, "_memory_record")
    assert value.lanes[0].st._eval_sid == 0
    assert value.broken is None


def test_admitted_rejects_geometry_outside_int32(monkeypatch):
    spec = valid_spec()
    value = decoder()
    monkeypatch.setattr(request_memory, "make_record",
                        lambda *args: {"effective_length": 2**40, "policy": {}, "spec": spec})
    request_memory.admitted(value, {"eval": spec, "sid": 0, "prompt": [1], "count": 1})
    assert "int32" in value.w.meta[KEY]["active"][0]["admission_error"]
    value = decoder()
    value.lanes[0].st.capacity = 2**40
    monkeypatch.setattr(request_memory, "make_record",
                        lambda *args: {"effective_length": 8, "policy": {}, "spec": spec})
    request_memory.admitted(value, {"eval": spec, "sid": 0, "prompt": [1], "count": 1})
    assert "int32" in value.w.meta[KEY]["active"][0]["admission_error"]
    assert value.broken is None


def test_missing_lane_stays_on_the_request():
    value = decoder()
    admission = {"eval": {"tag": "glm-eval-v1"}, "sid": 3, "prompt": [1], "count": 1}
    request_memory.admitted(value, admission)
    assert value.broken is None
    record = value.w.meta[KEY]["active"][3]
    assert record["admission_error"]
    assert record["effective_length"] == 0


@given(
    limit=st.integers(min_value=1, max_value=4000),
    payload=st.binary(max_size=500),
)
def test_limited_writer_never_passes_the_reservation(limit, payload):
    buffer = io.BytesIO()
    writer = LimitedWriter(buffer, limit)
    try:
        written = writer.write(payload)
    except ValueError as error:
        assert "reservation" in str(error)
        assert buffer.tell() <= limit
    else:
        assert written == len(payload)
        assert buffer.tell() == len(payload)
        assert buffer.tell() <= limit
