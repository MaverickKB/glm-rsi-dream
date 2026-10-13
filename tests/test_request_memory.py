"""CPU regressions for extent growth, paired rejection, and broken health."""
import ast
import gzip
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold_dream import attention_native, bootstrap, request_memory

KEY = request_memory.KEY


def decoder(rank=0, capacity=2048):
    stream = SimpleNamespace(prompt=[1, 2, 3], count=4096)
    slot = SimpleNamespace(capacity=capacity, reset=lambda: None)
    lane = SimpleNamespace(s=stream, st=slot)
    state = {"active": {}, "prefix": {"normal_ids": [1, 2]},
             "snapshot": SimpleNamespace(ids=[0] * 4096), "pair_sha256": "a" * 64}
    value = SimpleNamespace(w=SimpleNamespace(rank=rank, meta={KEY: state}),
                            g=SimpleNamespace(limit=1048576), lanes={0: lane},
                            rows_max=32, count=4, broken=None)
    value._need = lambda length: ((length + value.rows_max + 2047) // 2048) * 2048

    def finish(sid):
        value.lanes.pop(sid)
        state["active"].pop(sid)

    value._finish = finish
    return value


def native_method(module_name, class_name, method_name):
    sources = json.loads(gzip.decompress(
        (Path(__file__).parent / "upstream_sources.json.gz").read_bytes()))
    relative = "families/glm5_next/cuda/multi.py"
    if module_name == "cuda_http":
        relative = "cuda/http.py"
    source = bootstrap.transform(module_name, sources[relative])
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == method_name:
                    namespace = {"_memory": request_memory, "ALIGN": 2048, "Stream": SimpleNamespace}
                    exec(compile(ast.Module(body=[method], type_ignores=[]), relative, "exec"), namespace)
                    return namespace[method_name]
    raise AssertionError("native method missing")


@pytest.mark.parametrize("effective", [96, 224, 2047, 2048, 4096, 65536, 1040000])
def test_reserves_effective_prompt_and_native_growth(monkeypatch, effective):
    value = decoder()
    stream = value.lanes[0].s
    stream._memory_record = {"effective_length": effective}
    stream.glm = {"eval": {"arm": "main"}}
    need = request_memory.extent_need(value, stream)
    assert need >= effective + value.rows_max
    assert need == value._need(max(effective, len(stream.prompt)))


def test_untagged_reservation_is_native():
    value = decoder()
    stream = value.lanes[0].s
    assert request_memory.extent_need(value, stream) == value._need(len(stream.prompt))


def test_full_memory_is_reserved_before_native_admission(tmp_path, monkeypatch):
    value = decoder()
    state = value.w.meta[KEY]
    state['root'] = tmp_path
    (tmp_path / 'requests').mkdir()
    (tmp_path / 'requests/cpu-full.json').write_text('{"use_memory": true}')
    spec = {'tag': 'glm-eval-v1', 'request_id': 'cpu-full',
            'arm': 'main', 'pair_sha256': state['pair_sha256']}
    stream = value.lanes[0].s
    stream.glm = {'eval': spec}
    stream.count = 32768
    decode = SimpleNamespace(load_rows=lambda *args: None, restore=lambda *args: None)
    monkeypatch.setitem(sys.modules, 'tensorfold.families.glm5_next.cuda',
                        SimpleNamespace(decode=decode))
    record = request_memory.prepare_request(value, stream)
    assert record['effective_length'] == 4097
    assert stream.prompt == [1, 2, 3]
    assert state['active'] == {}
    capacity = request_memory.extent_need(value, stream)
    assert capacity == 6144
    assert capacity < stream.count
    value.lanes[0].st.capacity = capacity
    value.e = SimpleNamespace(use=lambda slot: None)
    record['sid'] = 0
    attention_native.install(value, record)
    assert len(stream.prompt) == 4097
    assert record['memory_activated']


def test_memory_context_overflow_rejects_before_rank_mutation(monkeypatch):
    value = decoder()
    stream = value.lanes[0].s
    stream.glm = {'eval': {'tag': 'glm-eval-v1', 'request_id': 'cpu-limit',
                  'arm': 'main', 'pair_sha256': 'a' * 64}}
    monkeypatch.setattr(request_memory, 'make_record',
                        lambda *args: {'effective_length': value.g.limit})
    with pytest.raises(ValueError, match='engine context limit'):
        request_memory.prepare_request(value, stream)
    assert stream.prompt == [1, 2, 3]
    assert value.w.meta[KEY]['active'] == {}
    assert value.broken is None


def test_bad_preflight_reaches_request_error_even_when_other_lanes_exist(monkeypatch):
    value = decoder()
    value.pool = SimpleNamespace(rows=8192, free_rows=lambda: 0, extents=[])
    stream = value.lanes[0].s

    def reject(*args):
        raise ValueError("invalid memory artifact")

    monkeypatch.setattr(request_memory, "prepare_request", reject)
    fits = native_method("multi", "MultiDecoder", "fits")
    assert fits(value, stream)
    with pytest.raises(ValueError, match="invalid memory artifact"):
        request_memory.extent_need(value, stream)
    assert value.broken is None


def test_install_accepts_large_reply_in_initial_small_extent(monkeypatch):
    value = decoder()
    snap = SimpleNamespace(ids=[0] * 96)
    calls = []
    decode = SimpleNamespace(load_rows=lambda *args: calls.append("rows"),
                             restore=lambda *args: calls.append("restore"))
    monkeypatch.setitem(sys.modules, "tensorfold.families.glm5_next.cuda",
                        SimpleNamespace(decode=decode))
    value.e = SimpleNamespace(use=lambda slot: None)
    record = {"spec": {"arm": "main"}, "policy": {"use_memory": True},
              "sid": 0, "memory_plan": (snap, [3])}
    attention_native.install(value, record)
    assert calls == ["rows", "restore"]
    assert value.lanes[0].s.count == 4096
    assert record["memory_activated"]
    assert len(value.lanes[0].s.prompt) == 97


@pytest.mark.parametrize("failure_rank", [None, 0, 1])
@pytest.mark.parametrize("failure_stage", ["preparation", "installation", "geometry"])
def test_pair_agrees_and_only_releases_failed_request(monkeypatch, failure_rank, failure_stage):
    pair = [decoder(0), decoder(1)]
    barrier = threading.Barrier(2)
    votes = {}

    def gather(rank, words):
        votes[rank] = words
        barrier.wait(timeout=5)
        result = [list(votes[0]), list(votes[1])]
        barrier.wait(timeout=5)
        return result

    for value in pair:
        rank = value.w.rank
        value.g._gather_ints = lambda words, rank=rank: gather(rank, words)
        value.lanes[1] = SimpleNamespace(s=SimpleNamespace(count=1))
        record = {"sid": 0, "effective_length": 97}
        if failure_rank == rank and failure_stage == "preparation":
            record["admission_error"] = "worker artifact unavailable"
        if failure_rank == rank and failure_stage == "geometry":
            record["effective_length"] += 32
        value.w.meta[KEY]["active"][0] = record

    def install(value, record):
        if failure_stage == "installation" and value.w.rank == failure_rank:
            raise ValueError("slot restore failed")
        record["memory_activated"] = True

    monkeypatch.setattr(attention_native, "install", install)
    with ThreadPoolExecutor(max_workers=2) as executor:
        head = executor.submit(request_memory.complete_admission, pair[0], 0)
        worker = executor.submit(request_memory.complete_admission, pair[1], 0)
        if failure_rank is None:
            head.result(timeout=10)
        else:
            with pytest.raises(ValueError, match="rejected this request"):
                head.result(timeout=10)
        worker.result(timeout=10)
    for value in pair:
        assert value.broken is None
        assert 1 in value.lanes
        assert (0 in value.lanes) == (failure_rank is None)
        assert (0 in value.w.meta[KEY]["active"]) == (failure_rank is None)


def test_rank_desync_health_returns_503():
    value = decoder()
    value.lanes = {}
    value.pool = SimpleNamespace(rows=8192, free_rows=lambda: 8192)
    value.kept = []
    value._held_bytes = lambda: 0
    value.kept_bytes_cap = 0
    value.group = False
    value.disk = None
    health = native_method("multi", "MultiDecoder", "health")
    handler = native_method("cuda_http", "Handler", "do_GET")
    handler.__globals__["health"] = SimpleNamespace(
        of=lambda app: SimpleNamespace(snapshot=lambda app: health(value)))
    handler.__globals__["app"] = None
    responses = []
    http = SimpleNamespace(path="/health", _json=lambda status, body: responses.append((status, body)))
    handler(http)
    assert responses[-1][0] == 200
    value.broken = RuntimeError("rank state diverged")
    handler(http)
    assert responses[-1][0] == 503
    assert responses[-1][1]["rank_desynchronized"]
    assert not responses[-1][1]["ok"]


def test_failed_rank_exchange_marks_decoder_broken():
    value = decoder()
    failure = RuntimeError('rank exchange failed')

    def gather(words):
        raise failure

    value.g._gather_ints = gather
    with pytest.raises(RuntimeError, match='rank exchange failed'):
        request_memory.gather_status(value, [1])
    assert value.broken is failure


class RowBuffer:
    def __init__(self, values):
        self.values = list(values)

    def fill_(self, number):
        self.values = [number for _ in self.values]

    def zero_(self):
        self.values = [0 for _ in self.values]


def test_rank_exchange_failure_releases_slot_and_clears_installed_rows(monkeypatch):
    value = decoder()
    rows = RowBuffer([1, 1, 1])
    value.lanes[0].st.kc = [rows]
    value.w.meta[KEY]["active"][0] = {"sid": 0, "effective_length": 97}
    calls = {"count": 0}

    def gather(words):
        calls["count"] += 1
        if calls["count"] == 1:
            return [list(words), list(words)]
        raise RuntimeError("rank exchange failed")

    def install(decoder_value, record):
        decoder_value.lanes[0].st.kc[0].fill_(4)
        record["memory_activated"] = True

    value.g._gather_ints = gather
    monkeypatch.setattr(attention_native, "install", install)
    with pytest.raises(RuntimeError, match="rank exchange failed"):
        request_memory.complete_admission(value, 0)
    assert value.broken is not None
    assert 0 not in value.lanes
    assert rows.values == [0, 0, 0]


def test_agreed_install_keeps_attention_rows(monkeypatch):
    value = decoder()
    rows = RowBuffer([0, 0])
    value.lanes[0].st.kc = [rows]
    value.w.meta[KEY]["active"][0] = {"sid": 0, "effective_length": 97}

    def gather(words):
        return [list(words), list(words)]

    def install(decoder_value, record):
        decoder_value.lanes[0].st.kc[0].fill_(4)
        record["memory_activated"] = True

    value.g._gather_ints = gather
    monkeypatch.setattr(attention_native, "install", install)
    request_memory.complete_admission(value, 0)
    assert rows.values == [4, 4]
    assert 0 in value.lanes


def load_follow():
    sources = json.loads(gzip.decompress(
        (Path(__file__).parent / "upstream_sources.json.gz").read_bytes()))
    source = bootstrap.transform("multi", sources["families/glm5_next/cuda/multi.py"])
    assert "_memory.unblock_peer(self)" in source
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == "follow":
            namespace = {
                "_memory": request_memory,
                "_watch": lambda *args, **kwargs: None,
                "_unwatch": lambda *args, **kwargs: None,
                "unseal": lambda msg, seq, rank=1: msg,
                "SEAL_MOD": 2_147_483_647,
                "torch": SimpleNamespace(no_grad=lambda: (lambda fn: fn)),
            }
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, "multi.py", "exec"), namespace)
            return namespace["follow"]
    raise AssertionError("follow missing")


def test_rank1_follow_unblocks_a_peer_blocked_in_admission(monkeypatch):
    pair = [decoder(0), decoder(1)]
    barrier = threading.Barrier(2)
    votes = {}

    def gather(rank, words):
        votes[rank] = list(words)
        barrier.wait(timeout=5)
        ordered = [list(votes[0]), list(votes[1])]
        barrier.wait(timeout=5)
        return ordered

    for value in pair:
        rank = value.w.rank
        value.g._gather_ints = lambda words, rank=rank: gather(rank, words)
        value.g._share = lambda values: [1, 2]
        value.received = 0
        value.idle = False
        value.rank = rank
        value.watchdog = None
        value.w.meta[KEY]["active"][0] = {"sid": 0, "effective_length": 97}

    def apply(msg):
        pair[1]._eval_admit = 0
        raise RuntimeError("rank 1 admit failed before the vote")

    def refuse_install(*args, **kwargs):
        raise AssertionError("installed")

    pair[1].apply = apply
    monkeypatch.setattr(attention_native, "install", refuse_install)
    follow = load_follow()
    with ThreadPoolExecutor(max_workers=2) as executor:
        head = executor.submit(request_memory.complete_admission, pair[0], 0)
        worker = executor.submit(follow, pair[1], True)
        with pytest.raises(ValueError, match="rejected this request"):
            head.result(timeout=5)
        worker.result(timeout=5)
    assert "admit failed" in str(pair[1].broken)
    assert votes[1] == [0, 0, 0, 0]
    assert 0 not in pair[0].lanes
    assert 0 not in pair[1].lanes


def test_rank1_follow_does_not_gather_again_after_an_aborted_vote():
    pair = [decoder(0), decoder(1)]
    barrier = threading.Barrier(2)
    calls = {0: 0, 1: 0}

    def gather(rank, words):
        calls[rank] += 1
        barrier.wait(timeout=5)
        raise RuntimeError("rank exchange failed")

    for value in pair:
        rank = value.w.rank
        value.g._gather_ints = lambda words, rank=rank: gather(rank, words)
        value.g._share = lambda values: [1, 2]
        value.received = 0
        value.idle = False
        value.rank = rank
        value.watchdog = None
        value.w.meta[KEY]["active"][0] = {"sid": 0, "effective_length": 97}

    def apply(msg):
        pair[1]._eval_admit = 0
        request_memory.complete_admission(pair[1], 0)

    pair[1].apply = apply
    follow = load_follow()
    with ThreadPoolExecutor(max_workers=2) as executor:
        head = executor.submit(request_memory.complete_admission, pair[0], 0)
        worker = executor.submit(follow, pair[1], True)
        with pytest.raises(RuntimeError, match="rank exchange failed"):
            head.result(timeout=5)
        worker.result(timeout=5)
    assert calls == {0: 1, 1: 1}
    assert 0 not in pair[0].lanes
    assert 0 not in pair[1].lanes
