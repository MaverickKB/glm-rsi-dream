"""Repeat the numerical parity with retained synthetic artifacts and a JSON receipt."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from dream_loader import bootstrap, runtime
from dream_loader.artifact import load_pair
from dream_loader.bundle import bundle


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    config = {"text_config": {"hidden_size": 4096, "moe_intermediate_size": 2048,
                              "n_shared_experts": 1, "n_routed_experts": 288,
                              "num_hidden_layers": 45, "first_k_dense_replace": 3}}
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config))
    generator = torch.Generator().manual_seed(98751)
    paths = []
    for rank in (0, 1):
        sites = {}
        for index in (11, 23, 35):
            sites[index] = {"gate": torch.randn(32, 4096, generator=generator) * 0.25,
                            "up": torch.randn(32, 4096, generator=generator) * 0.25,
                            "down": torch.randn(4096, 32, generator=generator) * 0.01}
        value = {"schema_version": 1, "mechanism": "shared_ffn_swiglu_v1", "rank": rank,
                 "world": 2, "base_frozen": True, "layers": sites}
        path = output / f"reference-rank{rank}.pt"
        torch.save(value, path)
        paths.append(path)
    manifest = bundle(paths[0], paths[1], config_path, output / "addition")
    path = output / "addition/manifest.json"
    _, pair = load_pair(path)
    test_path = Path(__file__).with_name("test_parity.py")
    spec = importlib.util.spec_from_file_location("loader_parity", test_path)
    parity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parity)
    native = parity.native_reference()
    cases = []
    for rank in (0, 1):
        for rows, prefill, chunked, mode in ((1, False, False, "1"),
                                            (7, False, True, "0"), (65, True, True, "1")):
            inputs = torch.randn(rows, 4096, generator=generator).bfloat16()
            base = torch.randn(rows, 4096, generator=generator)
            experts = torch.randn(rows, 3, 4096, generator=generator)
            cfg = SimpleNamespace(hidden=4096, limit=10, top_k=2, experts=288,
                                  routed_scale=1, norm_topk=True)
            w = SimpleNamespace(rank=rank, world=2, device="cpu", cfg=cfg,
                                meta={runtime.KEY: {"layers": pair[rank], "mechanism": manifest["binding"]["mechanism"]},
                                      native.KEY: {"root": path.parent}})
            record = {"policy": {"neurons": manifest["binding"]["ranks"][rank],
                                 "use_memory": False}}
            native.records = lambda w, b: [(record, None, 0, rows)]
            for index in (11, 23, 35, 19):
                layer = SimpleNamespace(index=index, moe=SimpleNamespace(shared=True,
                                                                         router=None, bias=None, experts=None))
                outputs = []
                with parity.pytest.MonkeyPatch.context() as patch:
                    patch.setenv("TF_GLM_COMBINE_SY", mode)
                    for function in (runtime.shared, native.shared):
                        b = SimpleNamespace(normed=inputs.clone(), sy=torch.zeros_like(base),
                                            part=torch.zeros_like(base), ey=experts.clone(),
                                            wts=torch.full((rows, 2), 0.5), mlog=torch.empty(rows, 288),
                                            pick=torch.zeros(rows, 2, dtype=torch.int32),
                                            plan=SimpleNamespace(tile=64), exl3=None, prefill=prefill)
                        result = parity.stock_moe(function, base, layer, w, b, rows, chunked=chunked)
                        outputs.append(result)
                equal = torch.equal(outputs[0], outputs[1])
                error = float((outputs[0] - outputs[1]).abs().max())
                cases.append({"rank": rank, "rows": rows, "prefill": prefill, "chunked": chunked,
                              "combine_sy": mode, "layer": index, "exact": equal, "max_abs": error})
                if not equal:
                    raise AssertionError("Same-artifact stock/DREAM parity failed")
    receipt = {"kind": "offline_cpu_inference_extension_parity", "synthetic_artifacts": True,
               "full_model_generation_tested": False, "cuda_tested": False,
               "hidden": 4096, "added_width_per_rank": 32, "world": 2,
               "pair_sha256": manifest["pair_sha256"], "artifacts": manifest["binding"]["ranks"],
               "dream_reference_commit": "e198c2d83d05759734afeb0793e9257873f57045",
               "dream_reference_sha256": parity.REFERENCE_SHA256,
               "stock_source_pins": bootstrap.PINS["engine_source_files"], "cases": cases}
    (output / "parity.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "all_exact": True, "max_abs": 0.0,
                      "report": str(output / "parity.json")}))


if __name__ == "__main__":
    run(sys.argv[1])
