"""CPU proofs for the shared-expert checkpoint transformation."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from glm_rsi_dream.export import FOUR_STREAMS, export_checkpoint, sha256
from glm_rsi_dream.files import digest


def shared(inputs, tensors, limit=10):
    gate = F.linear(inputs, tensors["gate"]).clamp(max=limit)
    up = F.linear(inputs, tensors["up"]).clamp(-limit, limit)
    return F.linear(F.silu(gate) * up, tensors["down"])


def fixture(tmp_path, dtype=torch.float32, model_prefix="model.language_model."):
    torch.manual_seed(47)
    base = tmp_path / "base"
    base.mkdir()
    text = {
        "model_type": "glm5_next_text", "hidden_size": 8, "moe_intermediate_size": 8,
        "n_shared_experts": 1, "n_routed_experts": 288, "num_hidden_layers": 3,
        "first_k_dense_replace": 1, "num_nextn_predict_layers": 1, "swiglu_limit": 10.0,
    }
    (base / "config.json").write_text(json.dumps({"model_type": "glm5_next", "text_config": text}))
    (base / "tokenizer.json").write_text('{"fixture": "fictional"}')
    weights = {}
    sites = {}
    for layer in (1, 2, 3):
        site = {"gate": torch.randn(8, 8).to(dtype), "up": torch.randn(8, 8).to(dtype),
                "down": torch.randn(8, 8).to(dtype)}
        sites[layer] = site
        for name, tensor in site.items():
            weights[f"{model_prefix}layers.{layer}.mlp.shared_experts.{name}_proj.weight"] = tensor
    marker = f"{model_prefix}layers.1.mlp.experts.0.gate_proj.trellis"
    weights[marker] = torch.arange(64, dtype=torch.int16).reshape(8, 8)
    weights[f"{model_prefix}embed_tokens.weight"] = torch.randn(4, 8).to(dtype)
    first = {}
    second = {}
    weight_map = {}
    for name, tensor in weights.items():
        if "down_proj" in name:
            second[name] = tensor
            weight_map[name] = "second.safetensors"
        else:
            first[name] = tensor
            weight_map[name] = "first.safetensors"
    save_file(first, base / "first.safetensors", metadata={"format": "pt"})
    save_file(second, base / "second.safetensors")
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    paths = []
    additions = []
    for rank in range(2):
        added = {}
        for layer in (1, 2):
            width = 2 + rank + (layer - 1)
            added[layer] = {"gate": torch.randn(width, 8), "up": torch.randn(width, 8),
                            "down": torch.randn(8, width)}
        artifact = {"schema_version": 1, "rank": rank, "world": 2, "base_frozen": True,
                    "mechanism": "shared_ffn_swiglu_v1", "layers": added}
        path = tmp_path / f"rank{rank}.pt"
        torch.save(artifact, path)
        paths.append(path)
        additions.append(added)
    return base, paths, sites, additions, marker


def read_weights(path):
    weights = {}
    for shard in Path(path).glob("*.safetensors"):
        weights.update(load_file(shard))
    return weights


@pytest.mark.parametrize("prefix", ["model.language_model.", "model."])
def test_merged_outputs_and_native_rank_order(tmp_path, prefix):
    base, paths, sites, additions, marker = fixture(tmp_path, model_prefix=prefix)
    original_hashes = {}
    for path in base.iterdir():
        original_hashes[path.name] = sha256(path)
    output = tmp_path / "merged"
    report = export_checkpoint(base, paths, output)
    weights = read_weights(output)
    inputs = torch.randn(43, 8) * 12  # Exercise both clamps, including negative gates.
    for layer in (1, 2, 3):
        actual = {}
        for name in ("gate", "up", "down"):
            key = f"{prefix}layers.{layer}.mlp.shared_experts.{name}_proj.weight"
            actual[name] = weights[key]
            axis = 1 if name == "down" else 0
            for rank in range(2):
                native = actual[name].narrow(axis, rank * 8, 8)
                frozen = sites[layer][name].narrow(axis, rank * 4, 4)
                assert torch.equal(native.narrow(axis, 0, 4), frozen)
                start = 4
                if layer in additions[rank]:
                    added = additions[rank][layer][name]
                    assert torch.equal(native.narrow(axis, start, added.shape[axis]), added)
                    start += added.shape[axis]
                assert torch.count_nonzero(native.narrow(axis, start, 8 - start)) == 0
        expected = shared(inputs, sites[layer])
        if layer != 3:
            expected += shared(inputs, additions[0][layer]) + shared(inputs, additions[1][layer])
        torch.testing.assert_close(shared(inputs, actual), expected, rtol=3e-5, atol=0.001)
    assert torch.equal(weights[marker], read_weights(base)[marker])
    assert report["new_shared_width"] == 16
    assert report["shared_mlp_branches_per_layer"] == 1
    assert report["routed_experts"] == 288
    assert report["stock_hosting_answer_verified"] is False
    assert (output / "tokenizer.json").read_bytes() == (base / "tokenizer.json").read_bytes()
    for path in base.iterdir():
        assert original_hashes[path.name] == sha256(path)
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(weights)
    assert index["metadata"]["total_size"] == sum(t.numel() * t.element_size() for t in weights.values())


def test_bf16_cast_error_is_reported(tmp_path):
    base, paths, _, _, _ = fixture(tmp_path, dtype=torch.bfloat16)
    report = export_checkpoint(base, paths, tmp_path / "merged")
    assert report["dtype_rounding_max"] > 0
    for tensor in read_weights(tmp_path / "merged").values():
        assert torch.isfinite(tensor).all()


@pytest.mark.parametrize("identical_blocks", [True, False])
def test_four_stream_stock_export_requires_exact_mapping(tmp_path, identical_blocks):
    base, paths, sites, additions, _ = fixture(tmp_path)
    for path in paths:
        artifact = torch.load(path, weights_only=True)
        artifact["mechanism"] = FOUR_STREAMS
        artifact["study_sha256"] = digest({"purpose": "synthetic CPU four-stream stock mapping"})
        for site in artifact["layers"].values():
            site["down"] = site["down"].repeat(4, 1)
            if not identical_blocks:
                site["down"][8:16] += 0.25
        torch.save(artifact, path)
    output = tmp_path / "merged"
    if not identical_blocks:
        with pytest.raises(ValueError, match="stock merge requires identical blocks"):
            export_checkpoint(base, paths, output)
        assert not output.exists()
        assert not list(tmp_path.glob("merged.partial-*"))
        return
    report = export_checkpoint(base, paths, output)
    assert report["addition_mechanism"] == FOUR_STREAMS
    assert report["four_stream_reduction"] == "identical_down_blocks"
    weights = read_weights(output)
    inputs = torch.randn(13, 8) * 12
    for layer in (1, 2):
        merged = {}
        for name in ("gate", "up", "down"):
            key = f"model.language_model.layers.{layer}.mlp.shared_experts.{name}_proj.weight"
            merged[name] = weights[key]
        expected = shared(inputs, sites[layer])
        for rank in (0, 1):
            expected += shared(inputs, additions[rank][layer])
        torch.testing.assert_close(shared(inputs, merged), expected, rtol=3e-5, atol=0.001)


def test_existing_output_and_source_overlap_are_rejected(tmp_path):
    base, paths, _, _, _ = fixture(tmp_path)
    for output in (base, base / "nested", tmp_path):
        with pytest.raises(ValueError):
            export_checkpoint(base, paths, output)


@pytest.mark.parametrize("problem", ["rank", "nonfinite", "layer", "hidden", "activation"])
def test_invalid_addition_is_rejected_before_output(tmp_path, problem):
    base, paths, _, _, _ = fixture(tmp_path)
    artifact = torch.load(paths[1], weights_only=True)
    if problem == "rank":
        artifact["rank"] = 0
    elif problem == "nonfinite":
        artifact["layers"][1]["down"][0, 0] = float("nan")
    elif problem == "layer":
        artifact["layers"][0] = artifact["layers"].pop(1)
    elif problem == "hidden":
        artifact["layers"][1]["gate"] = torch.ones(3, 9)
    else:
        artifact["swiglu_limit"] = 8
    torch.save(artifact, paths[1])
    with pytest.raises(ValueError):
        export_checkpoint(base, paths, tmp_path / "merged")
    assert not (tmp_path / "merged").exists()
    assert not list(tmp_path.glob("merged.partial-*"))


def test_invalid_base_cleans_partially_written_output(tmp_path):
    base, paths, _, _, _ = fixture(tmp_path)
    weights = load_file(base / "second.safetensors")
    name = next(iter(weights))
    weights[name][0, 0] = float("nan")
    save_file(weights, base / "second.safetensors")
    with pytest.raises(ValueError, match="base shared matrix"):
        export_checkpoint(base, paths, tmp_path / "merged")
    assert not (tmp_path / "merged").exists()
    assert not list(tmp_path.glob("merged.partial-*"))


def test_corrupt_index_is_rejected(tmp_path):
    base, paths, _, _, _ = fixture(tmp_path)
    (base / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    with pytest.raises(ValueError, match="index"):
        export_checkpoint(base, paths, tmp_path / "merged")


def test_mixed_exl3_configuration_and_tensor_bytes_survive(tmp_path):
    base, paths, _, _, marker = fixture(tmp_path, dtype=torch.bfloat16)
    config = json.loads((base / "config.json").read_text())
    config["quantization_config"] = {"quant_method": "exl3", "bits": 4, "codebook": "mcg",
                                      "scope": "glm53_routed_experts_only"}
    (base / "config.json").write_text(json.dumps(config))
    output = tmp_path / "mixed-export"
    report = export_checkpoint(base, paths, output)
    actual = json.loads((output / "config.json").read_text())
    assert actual["quantization_config"] == config["quantization_config"]
    assert torch.equal(read_weights(base)[marker], read_weights(output)[marker])
    assert report["source_quantization"]["quant_method"] == "exl3"
    assert report["fresh_exl3_quantization_performed"] is False
