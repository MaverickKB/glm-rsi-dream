"""CPU parity for four-stream additions, with no DREAM engine import."""

import argparse
import ast
import hashlib
import importlib
import json
import sys
import types
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from dream_loader import bootstrap, runtime
from dream_loader.artifact import FOUR_STREAMS, load_pair
from stock_cpu import FORWARD, SOURCES, stock_moe

REFERENCE = Path(__file__).with_name("dream_four_stream_reference.txt")
REFERENCE_SHA = "c704f3f7e4d64be5db3a02d0b6e37f102d0f6db0f6aee813d311e2ccd428d2ea"


def native_reference():
    raw = REFERENCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == REFERENCE_SHA
    module = types.ModuleType("frozen_four_stream_native")
    exec(compile(raw, str(REFERENCE), "exec"), module.__dict__)
    return module


def base_post(before, gathered, post, combine):
    base = gathered[0].float() + gathered[1].float()
    branch = base.to(before.dtype).float()
    streams = before.float()
    output = []
    for index in range(4):
        mixed = streams[:, 0] * combine[:, 0, index, None]
        for source in range(1, 4):
            mixed = mixed + streams[:, source] * combine[:, source, index, None]
        output.append((mixed + post[:, index, None] * branch).to(before.dtype))
    return torch.stack(output, dim=1)


def stock_ffn_update(hooks, layer, w, b, base, peer_base, *, chunked, forward=None):
    """Run stock layer_forward and moe_block with explicit CPU kernel doubles."""
    rows = len(base)
    posts = []

    def hc_post(out, inputs, gathered, post, combine):
        posts.append(gathered)
        if len(posts) == 1:
            return  # Attention is an explicit identity double for this FFN probe.
        before = inputs.reshape(rows, 4, w.cfg.hidden).clone()
        result = base_post(before, gathered, post, combine.reshape(rows, 4, 4))
        out.copy_(result.reshape_as(out))

    def ffn(layer, weights, buffers, count):
        partial = stock_moe(hooks.shared, base, layer, weights, buffers, count, chunked=chunked)
        if w.rank == 0:
            return torch.stack((partial, peer_base))
        return torch.stack((peer_base, partial))

    if forward is not None:
        exl3 = importlib.import_module(bootstrap.PREFIX + "exl3_mm")

        def front(layer, weights, buffers, lo, hi):
            buffers.sy[lo:hi].copy_(base[lo:hi])

        def partials(weights, buffers, count, fill, label=None, site=None):
            if chunked:
                middle = count // 2
                fill(0, middle)
                fill(middle, count)
            else:
                fill(0, count)
            partial = buffers.part[:count].clone()
            if w.rank == 0:
                return torch.stack((partial, peer_base))
            return torch.stack((peer_base, partial))

        def combine_shared(experts, shared, coefficients, out):
            out.copy_(shared + (experts[:, :2] * coefficients.unsqueeze(-1)).sum(1))

        def combine(experts, coefficients, out):
            out.copy_((experts[:, :2] * coefficients.unsqueeze(-1)).sum(1) + experts[:, -1])

        with ExitStack() as stack:
            stack.enter_context(patch.object(forward, "_dream", hooks))
            stack.enter_context(patch.object(forward, "_mixer", lambda *args: None))
            stack.enter_context(patch.object(forward.prof, "timed", lambda _: nullcontext()))
            stack.enter_context(patch.object(forward.glue, "hc_pre", lambda *args, **kwargs: None))
            stack.enter_context(patch.object(forward.glue, "hc_post", hc_post))
            stack.enter_context(patch.object(forward.glue, "router", lambda *args: None))
            stack.enter_context(patch.object(forward.glue, "select", lambda *args: None))
            stack.enter_context(patch.object(forward.glue, "combine_shared", combine_shared))
            stack.enter_context(patch.object(forward.glue, "combine", combine))
            stack.enter_context(patch.object(forward.grouped, "route", lambda *args: None))
            stack.enter_context(patch.object(exl3, "routed", lambda *args: None))
            stack.enter_context(patch.object(forward.l2pf, "ACTIVE", None))
            stack.enter_context(patch.object(forward, "shared_front", front))
            stack.enter_context(patch.object(forward, "partials", partials))
            forward.layer_forward(layer, w, None, b, rows)
        assert len(posts) == 2
        return b.x.clone()

    namespace = {"LayerW": object, "Weights": object, "State": object, "Buffers": object,
                 "torch": torch, "_dream": hooks, "_ffn": ffn,
                 "_mixer": lambda *args: None, "prof": SimpleNamespace(timed=lambda _: nullcontext()),
                 "glue": SimpleNamespace(hc_pre=lambda *args, **kwargs: None, hc_post=hc_post)}
    source = bootstrap.transform(FORWARD, SOURCES[bootstrap.MODULES[FORWARD]])
    function = None
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == "layer_forward":
            function = node
            break
    assert function is not None
    exec(compile(ast.Module(body=[function], type_ignores=[]), "stock-four-stream-layer", "exec"), namespace)
    namespace["layer_forward"](layer, w, None, b, rows)
    assert len(posts) == 2
    return b.x.clone()


def parity_case(path, rank, rows, prefill, chunked, mode, index, *, forward=None, expect_change=None):
    manifest, pair = load_pair(path)
    native = native_reference()
    generator = torch.Generator().manual_seed(984 + rows + index)
    inputs = torch.randn(rows, 4096, generator=generator).bfloat16()
    before = torch.randn(rows, 4, 4096, generator=generator).bfloat16()
    post = torch.rand(rows, 4, generator=generator) + 0.2
    combine = torch.rand(rows, 4, 4, generator=generator)
    base = torch.randn(rows, 4096, generator=generator)
    routed = torch.randn(rows, 3, 4096, generator=generator)
    peer_base = torch.randn(rows, 4096, generator=generator)
    cfg = SimpleNamespace(hidden=4096, streams=4, limit=10, top_k=2, experts=288,
                          routed_scale=1, norm_topk=True, eps=1e-5, hc_eps=1e-5, hc_iters=20)
    layer = SimpleNamespace(index=index, mlp=None, attn_hc=SimpleNamespace(fn=None, base=None, scale=None),
                            ffn_hc=SimpleNamespace(fn=None, base=None, scale=None),
                            in_norm=None, post_norm=None,
                            moe=SimpleNamespace(shared=True, router=None, bias=None, experts=None))
    record = {"policy": {"neurons": manifest["binding"]["ranks"][rank], "use_memory": False}}
    native.records = lambda w, b: [(record, None, 0, rows)]
    peer = 1 - rank
    peer_site = pair[peer].get(index)
    peer_delta = torch.zeros(rows, 4, 4096)
    if peer_site is not None:
        peer_w = SimpleNamespace(rank=peer, world=2, device="cpu", cfg=cfg,
                                 meta={native.KEY: {"root": path.parent}})
        peer_record = {"policy": {"neurons": manifest["binding"]["ranks"][peer]}}
        native.records = lambda w, b: [(peer_record, None, 0, rows)]
        peer_b = SimpleNamespace(normed=inputs.clone(), x=before.reshape(rows, -1).clone(), prefill=False)
        native.before_ffn(layer, peer_w, peer_b, rows)
        native.shared(layer, peer_w, peer_b, 0, rows)
        peer_delta = peer_b._dream_delta.clone()
        native.records = lambda w, b: [(record, None, 0, rows)]

    class Communicator:
        def all_gather(self, partial, output):
            assert partial.ndim == 1 and output.ndim == 1
            assert partial.is_contiguous() and output.is_contiguous()
            assert output.numel() == 2 * partial.numel()
            paired = output.reshape(2, rows, 4, 4096)
            paired[rank].copy_(partial.reshape(rows, 4, 4096))
            paired[peer].copy_(peer_delta)

    w = SimpleNamespace(rank=rank, world=2, device="cpu", cfg=cfg, comm=Communicator(),
                        meta={runtime.KEY: {"layers": pair[rank], "mechanism": FOUR_STREAMS},
                              native.KEY: {"root": path.parent}})
    with patch.dict("os.environ", {"TF_GLM_COMBINE_SY": mode}):
        outputs = []
        for hooks in (runtime, native):
            b = SimpleNamespace(normed=inputs.clone(), x=before.reshape(rows, -1).clone(),
                                post=post.clone(), comb=combine.clone(), sy=torch.zeros_like(base),
                                part=torch.zeros_like(base), ey=routed.clone(), wts=torch.full((rows, 2), 0.5),
                                mlog=torch.empty(rows, 288), pick=torch.zeros(rows, 2, dtype=torch.int32),
                                plan=SimpleNamespace(tile=64), exl3=None, prefill=prefill,
                                xs=torch.empty(rows, 1), hcpart=torch.empty(rows, 1))
            outputs.append(stock_ffn_update(hooks, layer, w, b, base, peer_base,
                                            chunked=chunked, forward=forward))
            assert b._dream_delta is None and b._dream_before is None
    assert torch.equal(outputs[0], outputs[1])
    local_base = base + (routed[:, :2] * 0.5).sum(1)
    gathered = torch.stack((local_base, peer_base)) if rank == 0 else torch.stack((peer_base, local_base))
    baseline = base_post(before, gathered, post, combine).reshape_as(outputs[0])
    streams_changed = []
    for stream in range(4):
        actual = outputs[0].reshape(rows, 4, 4096)[:, stream]
        plain = baseline.reshape(rows, 4, 4096)[:, stream]
        streams_changed.append(not torch.equal(actual, plain))
    if expect_change is None:
        expect_change = index in pair[rank]
    assert streams_changed == [expect_change] * 4
    return {"rank": rank, "layer": index, "rows": rows, "prefill": prefill,
            "chunked": chunked, "combine_sy": mode, "exact": True,
            "max_abs": float((outputs[0].float() - outputs[1].float()).abs().max()),
            "streams_changed": streams_changed, "pair_sha256": manifest["pair_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--stock-imported", action="store_true")
    parser.add_argument("--artifact-origin", default="unclassified")
    args = parser.parse_args()
    torch.set_num_threads(1)
    forward = None
    if args.stock_imported:
        bootstrap.install()
        forward = importlib.import_module(FORWARD)
        importlib.import_module(bootstrap.PREFIX + "engine")
    cases = []
    for rank in (0, 1):
        for rows, prefill, chunked, mode in ((1, False, False, "1"),
                                            (7, False, True, "0"), (65, True, True, "1")):
            for index in (11, 23, 35, 19):
                cases.append(parity_case(args.manifest, rank, rows, prefill, chunked, mode,
                                         index, forward=forward))
    assert not any(name.startswith(("glm_rsi_dream", "tensorfold_dream")) for name in sys.modules)
    receipt = {"kind": "four_stream_stock_cpu_parity", "artifact_origin": args.artifact_origin,
               "synthetic_inputs": True,
               "cuda_used": False, "model_loaded": False, "api_started": False,
               "stock_modules_imported": args.stock_imported,
               "stock_source_profile": bootstrap.PINS["engine_source_files"],
               "reference_sha256": REFERENCE_SHA, "cases": cases,
               "backend": "explicit CPU doubles for attention, normalization, base shared/routed and TP exchange"}
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
