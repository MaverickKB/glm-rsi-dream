"""CPU parity through imported stock TensorFold modules in the derived image."""

import argparse
import hashlib
import importlib
import io
import json
import sys
import types
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from dream_loader import bootstrap, runtime
from dream_loader.artifact import load_pair

REFERENCE_SHA = "305a4abc311a0f3ba32b47b4fabe7acbc869f9bcc4efab2083c3162ebca0ece8"


def captured_inputs(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    inputs = []
    for rank, selected in enumerate(manifest["ranks"]):
        raw = (path.parent / selected["path"]).read_bytes()
        assert selected["rank"] == rank
        assert hashlib.sha256(raw).hexdigest() == selected["sha256"]
        capture = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
        assert capture["schema_version"] == 1 and capture["rank"] == rank
        assert capture["world"] == 2 and capture["attention_memory"] is False
        layers = {}
        for index, pieces in capture["layers"].items():
            rows = []
            for piece in pieces:
                rows.append(piece["input"])
            tensor = torch.cat(rows)
            assert tensor.ndim == 2 and tensor.shape[1] == 4096
            assert len(tensor) > 0 and torch.isfinite(tensor).all()
            layers[index] = tensor
        inputs.append(layers)
    assert len(inputs) == 2
    return manifest, inputs


def run(manifest_path, reference_path, reference_sha=REFERENCE_SHA,
        capture_manifest=None, artifact_origin="unclassified"):
    torch.set_num_threads(1)
    reference_path = Path(reference_path)
    raw = reference_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == reference_sha
    reference = types.ModuleType("dream_native_reference")
    exec(compile(raw, str(reference_path), "exec"), reference.__dict__)
    bootstrap.install()
    forward = importlib.import_module(bootstrap.PREFIX + "forward")
    engine = importlib.import_module(bootstrap.PREFIX + "engine")
    exl3 = importlib.import_module(bootstrap.PREFIX + "exl3_mm")
    manifest_path = Path(manifest_path)
    manifest, pair = load_pair(manifest_path)
    capture_identity = None
    native_inputs = None
    if capture_manifest is not None:
        capture_identity, native_inputs = captured_inputs(capture_manifest)
    indices = sorted(pair[0])
    if native_inputs is not None:
        indices = sorted(native_inputs[0])
        assert set(pair[0]).issubset(indices)
        assert set(indices) == set(native_inputs[1])
    if 19 not in indices:
        indices.append(19)
    generator = torch.Generator().manual_seed(8271)
    cases = []

    def combine_shared(experts, shared, coefficients, out):
        routed = (experts[:, :2] * coefficients.unsqueeze(-1)).sum(1)
        out.copy_(shared + routed)

    def combine(experts, coefficients, out):
        routed = (experts[:, :2] * coefficients.unsqueeze(-1)).sum(1)
        out.copy_(routed + experts[:, -1])

    for rank in (0, 1):
        for requested_rows, prefill, chunked, mode in ((1, False, False, "1"),
                                                      (7, False, True, "0"), (65, True, True, "1")):
            cfg = SimpleNamespace(hidden=4096, limit=10, top_k=2, experts=288,
                                  routed_scale=1, norm_topk=True)
            w = SimpleNamespace(rank=rank, device="cpu", cfg=cfg,
                                meta={runtime.KEY: {"layers": pair[rank], "mechanism": manifest["binding"]["mechanism"]},
                                      reference.KEY: {"root": manifest_path.parent}})
            record = {"policy": {"neurons": manifest["binding"]["ranks"][rank],
                                 "use_memory": False}}
            reference.records = lambda w, b: [(record, None, 0, rows)]

            def shared_front(layer, weights, buffers, lo, hi):
                buffers.sy[lo:hi].copy_(base[lo:hi])

            def partials(weights, buffers, count, fill, label, site):
                if chunked:
                    middle = count // 2
                    fill(0, middle)
                    fill(middle, count)
                else:
                    fill(0, count)
                return buffers.part[:count].clone()

            with ExitStack() as stack:
                stack.enter_context(patch.dict("os.environ", {"TF_GLM_COMBINE_SY": mode}))
                stack.enter_context(patch.object(forward.prof, "timed", lambda label: nullcontext()))
                stack.enter_context(patch.object(forward.glue, "router", lambda *args: None))
                stack.enter_context(patch.object(forward.glue, "select", lambda *args: None))
                stack.enter_context(patch.object(forward.glue, "combine_shared", combine_shared))
                stack.enter_context(patch.object(forward.glue, "combine", combine))
                stack.enter_context(patch.object(forward.grouped, "route", lambda *args: None))
                stack.enter_context(patch.object(exl3, "routed", lambda *args: None))
                stack.enter_context(patch.object(forward.l2pf, "ACTIVE", None))
                stack.enter_context(patch.object(forward, "shared_front", shared_front))
                stack.enter_context(patch.object(forward, "partials", partials))
                for index in indices:
                    input_layer = index
                    if native_inputs is None:
                        inputs = torch.randn(requested_rows, 4096, generator=generator).bfloat16()
                    else:
                        if index not in native_inputs[rank]:
                            input_layer = sorted(native_inputs[rank])[0]
                        captured = native_inputs[rank][input_layer]
                        inputs = captured[:requested_rows].clone()
                    rows = len(inputs)
                    base = torch.randn(rows, 4096, generator=generator)
                    experts = torch.randn(rows, 3, 4096, generator=generator)
                    layer = SimpleNamespace(index=index, moe=SimpleNamespace(shared=True,
                                                                             router=None, bias=None, experts=None))
                    outputs = []
                    for function in (runtime.shared, reference.shared):
                        b = SimpleNamespace(normed=inputs.clone(), sy=torch.zeros_like(base),
                                            part=torch.zeros_like(base), ey=experts.clone(),
                                            wts=torch.full((rows, 2), 0.5), mlog=torch.empty(rows, 288),
                                            pick=torch.zeros(rows, 2, dtype=torch.int32),
                                            plan=SimpleNamespace(tile=64), exl3=None, prefill=prefill)
                        with patch.object(forward._dream, "shared", function):
                            result = forward.moe_block(layer, w, b, rows)
                        outputs.append(result)
                    equal = torch.equal(outputs[0], outputs[1])
                    error = float((outputs[0] - outputs[1]).abs().max())
                    assert equal, "Stock module and native engine correction differ"
                    baseline = base + (experts[:, :2] * 0.5).sum(1)
                    cases.append({"rank": rank, "layer": index, "rows": rows,
                                  "input_layer": input_layer,
                                  "prefill": prefill, "chunked": chunked,
                                  "combine_sy": mode, "exact": equal, "max_abs": error,
                                  "correction_max_abs": float((outputs[0] - baseline).abs().max())})
    assert not any(name.startswith(("glm_rsi_dream", "tensorfold_dream")) for name in sys.modules)
    widths = {}
    for index, site in pair[0].items():
        widths[str(index)] = site["gate"].shape[0]
    receipt = {"kind": "stock_image_imported_moe_cpu_parity", "artifact_origin": artifact_origin,
               "cuda_used": False, "model_loaded": False, "api_started": False,
               "full_model_generation_tested": False, "hidden": 4096, "widths_per_rank": widths,
               "stock_forward_file": forward.__file__, "stock_engine_file": engine.__file__,
               "pair_sha256": manifest["pair_sha256"], "dream_reference_sha256": reference_sha,
               "capture_identity": capture_identity,
               "cuda_backend": "explicit CPU doubles for route, base shared, combine, TP reduction",
               "cases": cases}
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--reference-sha256", default=REFERENCE_SHA)
    parser.add_argument("--capture-manifest", type=Path)
    parser.add_argument("--artifact-origin", default="unclassified")
    args = parser.parse_args()
    run(args.manifest, args.reference, args.reference_sha256,
        args.capture_manifest, args.artifact_origin)
