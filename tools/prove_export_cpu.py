"""Reproduce CPU component evidence, never a native EXL3 hosting acceptance."""

import argparse
import ast
import importlib.util
import json
import runpy
from dataclasses import dataclass, field
from pathlib import Path

import torch
import transformers
from safetensors.torch import load_file, save_file
from torch.nn import functional as F
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMoE

from glm_rsi_dream.export import export_checkpoint, merged_matrix, sha256


def shared(inputs, site):
    gate = F.linear(inputs, site["gate"]).clamp(max=10)
    up = F.linear(inputs, site["up"]).clamp(-10, 10)
    return F.linear(F.silu(gate) * up, site["down"])


def stock_config_class(source):
    tree = ast.parse(source.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Config":
            selected.append(node)
    if len(selected) != 1:
        raise ValueError("Stock Config class identity changed")
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"dataclass": dataclass, "field": field, "Path": Path, "json": json}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["Config"]


def prove(output, evidence, trainer_path):
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(211)
    spec = importlib.util.spec_from_file_location("dream_neurons_fitter", trainer_path)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    inputs = torch.randn(64, 8)
    positive = torch.arange(64) < 32
    artifacts = []
    trained = []
    histories = []
    for rank in range(2):
        target_site = {"gate": torch.randn(2, 8) * 0.2, "up": torch.randn(2, 8) * 0.2,
                       "down": torch.randn(8, 2) * 0.2}
        targets = shared(inputs, target_site)
        targets[~positive] = 0
        site, history = trainer.fit_site(inputs, targets, positive, width=2, steps=200)
        artifact = {"schema_version": 1, "mechanism": "shared_ffn_swiglu_v1", "world": 2,
                    "rank": rank, "base_frozen": True, "layers": {3: site}}
        path = output / f"rank{rank}.pt"
        torch.save(artifact, path)
        artifacts.append(path)
        trained.append(site)
        histories.append(history)
    base = output / "base"
    base.mkdir()
    config = json.loads((evidence / "model-config.json").read_text())
    text = config["text_config"]
    text.update(hidden_size=8, moe_intermediate_size=8, num_hidden_layers=4,
                first_k_dense_replace=3, num_nextn_predict_layers=0,
                layer_types=["full_attention"] * 4,
                mlp_layer_types=["dense", "dense", "dense", "sparse"])
    config.pop("quantization_config")
    (base / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    original = {"gate": torch.randn(8, 8), "up": torch.randn(8, 8), "down": torch.randn(8, 8)}
    weights = {}
    prefix = "model.language_model.layers.3.mlp.shared_experts."
    for name, tensor in original.items():
        weights[prefix + name + "_proj.weight"] = tensor
    save_file(weights, base / "model.safetensors", metadata={"format": "pt"})
    merged = output / "merged"
    report = export_checkpoint(base, artifacts, merged)
    Config = stock_config_class(evidence / "stock-weights.py")
    stock_cfg = Config.read(merged)
    assert stock_cfg.shared_width == 16 and stock_cfg.moe_width == 8 and stock_cfg.experts == 288
    stock_split = runpy.run_path(str(evidence / "stock-split.py"))
    native_parts = []
    for rank in range(2):
        folder = output / f"stock-rank{rank}"
        stock_split["split_file"](merged / "model.safetensors", folder, rank)
        tensors = load_file(folder / f"model.rank{rank}.safetensors")
        site = {}
        for name in ("gate", "up", "down"):
            site[name] = tensors[prefix + name + "_proj.weight"]
        native_parts.append(site)
    expected = shared(inputs, original) + shared(inputs, trained[0]) + shared(inputs, trained[1])
    paired = shared(inputs, native_parts[0]) + shared(inputs, native_parts[1])
    torch.testing.assert_close(paired, expected, rtol=2e-5, atol=2e-5)
    updated = json.loads((merged / "config.json").read_text())
    stock_moe = Glm5NextTextMoE(Glm5NextTextConfig(**updated["text_config"]))
    stored = load_file(merged / "model.safetensors")
    shared_weights = {}
    for name, tensor in stored.items():
        shared_weights[name.removeprefix(prefix)] = tensor
    stock_moe.shared_experts.load_state_dict(shared_weights, strict=True)
    stock_output = stock_moe.shared_experts(inputs)
    torch.testing.assert_close(stock_output, expected, rtol=2e-5, atol=2e-5)

    # Actual hidden/base/addition geometry, CPU float32. One component, no model answer.
    real_inputs = torch.randn(3, 4096) * 0.1
    real_base = {"gate": torch.randn(2048, 4096) * 0.01,
                 "up": torch.randn(2048, 4096) * 0.01,
                 "down": torch.randn(4096, 2048) * 0.01}
    additions = []
    for rank in range(2):
        width = 32 if rank == 0 else 128
        additions.append({11: {"gate": torch.randn(width, 4096) * 0.01,
                              "up": torch.randn(width, 4096) * 0.01,
                              "down": torch.randn(4096, width) * 0.01}})
    real_merged = {}
    for name in ("gate", "up", "down"):
        real_merged[name], _ = merged_matrix(real_base[name], name, name, 11,
                                             additions, 2048, 4096, 4096)
    real_expected = shared(real_inputs, real_base)
    real_expected += shared(real_inputs, additions[0][11]) + shared(real_inputs, additions[1][11])
    real_actual = shared(real_inputs, real_merged)
    torch.testing.assert_close(real_actual, real_expected, rtol=1e-4, atol=1e-5)
    result = {
        "status": "CPU COMPONENT PASS", "trained_synthetic_addition": True,
        "trainer_source_sha256": sha256(trainer_path), "training": histories,
        "merge": report, "stock_config_class_sha256": sha256(evidence / "stock-weights.py"),
        "stock_split_source_sha256": sha256(evidence / "stock-split.py"),
        "stock_config_shared_width": stock_cfg.shared_width,
        "stock_config_routed_width": stock_cfg.moe_width,
        "stock_rank_split_max_abs_error": float((paired - expected).abs().max()),
        "stock_transformers_shared_mlp_max_abs_error": float((stock_output - expected).abs().max()),
        "stock_transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "real_geometry": {"hidden": 4096, "base_width": 2048, "added_rank_widths": [32, 128],
                          "merged_width": 4096, "max_abs_error": float((real_actual - real_expected).abs().max())},
        "full_model_checkpoint": False, "exl3_conversion_executed": False,
        "stock_tensorfold_model_load": False, "native_generated_answer": False,
        "device": "cpu",
    }
    (output / "proof.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    args = parser.parse_args()
    prove(args.output, args.evidence, args.trainer)


if __name__ == "__main__":
    main()
