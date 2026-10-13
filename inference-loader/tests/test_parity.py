"""Same artifact through the frozen DREAM native function and stock MoE call site."""

import hashlib
import json
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from dream_loader import runtime
from dream_loader.artifact import load_pair
from stock_cpu import stock_moe

REFERENCE = Path(__file__).with_name("dream_native_reference.txt")
REFERENCE_SHA256 = '305a4abc311a0f3ba32b47b4fabe7acbc869f9bcc4efab2083c3162ebca0ece8'


def native_reference():
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == REFERENCE_SHA256
    module = types.ModuleType("frozen_dream_neurons_native")
    exec(compile(REFERENCE.read_text(), str(REFERENCE), "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("rows,prefill,chunked,combine_sy", [
    (1, False, False, "1"), (7, False, True, "0"), (65, True, True, "1")])
def test_same_artifact_stock_combine_matches_dream_native(addition, monkeypatch, rank,
                                                          rows, prefill, chunked, combine_sy):
    path, _, manifest = addition
    _, pair = load_pair(path)
    native = native_reference()
    generator = torch.Generator().manual_seed(418 + rank)
    inputs = torch.randn(rows, 4096, generator=generator).bfloat16()
    base = torch.randn(rows, 4096, generator=generator)
    routed = torch.randn(rows, 3, 4096, generator=generator)
    cfg = SimpleNamespace(hidden=4096, limit=10, top_k=2, experts=288,
                          routed_scale=1, norm_topk=True)
    weights = SimpleNamespace(rank=rank, world=2, device="cpu", cfg=cfg,
                              meta={runtime.KEY: {"layers": pair[rank], "mechanism": manifest["binding"]["mechanism"]},
                                    native.KEY: {"root": path.parent}})
    record = {"policy": {"neurons": manifest["binding"]["ranks"][rank], "use_memory": False}}
    native.records = lambda w, b: [(record, None, 0, rows)]
    monkeypatch.setenv("TF_GLM_COMBINE_SY", combine_sy)
    for index in (11, 23, 35, 19):
        layer = SimpleNamespace(index=index, moe=SimpleNamespace(shared=True, router=None,
                                                                bias=None, experts=None))
        outputs = []
        for function in (runtime.shared, native.shared):
            buffers = SimpleNamespace(normed=inputs.clone(), sy=torch.zeros_like(base),
                                      part=torch.zeros_like(base), ey=routed.clone(),
                                      wts=torch.full((rows, 2), 0.5), mlog=torch.empty(rows, 288),
                                      pick=torch.zeros(rows, 2, dtype=torch.int32), plan=SimpleNamespace(tile=64), exl3=None, prefill=prefill)
            outputs.append(stock_moe(function, base, layer, weights, buffers, rows, chunked=chunked))
        assert torch.equal(outputs[0], outputs[1]), "Stock combine must match the engine exactly"
        if index != 19:
            baseline = base + (routed[:, :2] * 0.5).sum(1)
            assert not torch.equal(outputs[0], baseline), "Exercise a nonzero shared addition"
        else:
            assert torch.equal(outputs[0], base + (routed[:, :2] * 0.5).sum(1))
    assert record["neuron_sha256"] == manifest["binding"]["ranks"][rank]["sha256"]
    assert record["policy"]["use_memory"] is False


def test_pair_reduction_and_graph_safe_fixed_weights(addition):
    path, _, manifest = addition
    _, pair = load_pair(path)
    native = native_reference()
    generator = torch.Generator().manual_seed(914)
    inputs = torch.randn(4, 4096, generator=generator).bfloat16()
    plugin = []
    engine = []
    for rank in range(2):
        cfg = SimpleNamespace(hidden=4096, limit=10)
        w = SimpleNamespace(rank=rank, device="cpu", cfg=cfg,
                            meta={runtime.KEY: {"layers": pair[rank], "mechanism": manifest["binding"]["mechanism"]},
                                  native.KEY: {"root": path.parent}})
        raw = json.loads(path.read_text())["binding"]["ranks"][rank]
        record = {"policy": {"neurons": raw}}
        native.records = lambda w, b: [(record, None, 0, 4)]
        outputs = []
        for function in (runtime.shared, native.shared):
            b = SimpleNamespace(normed=inputs.clone(), sy=torch.zeros(4, 4096))
            function(SimpleNamespace(index=11), w, b, 0, 4)
            outputs.append(b.sy.clone())
        plugin.append(outputs[0])
        engine.append(outputs[1])
        for site in pair[rank].values():
            for tensor in site.values():
                assert tensor.requires_grad is False
        # The hot path never reads a file, reaches a network, or changes policy.
        original_bytes = path.read_bytes()
        path.unlink()
        b = SimpleNamespace(normed=inputs.clone(), sy=torch.zeros(4, 4096))
        runtime.shared(SimpleNamespace(index=11), w, b, 0, 4)
        assert torch.equal(b.sy, outputs[0])
        path.write_bytes(original_bytes)
    assert torch.equal(plugin[0] + plugin[1], engine[0] + engine[1])
