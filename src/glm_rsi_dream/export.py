"""Bake paired added SwiGLU neurons into ordinary GLM safetensors weights."""

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save

SHARED = re.compile(
    r"^(model(?:\.language_model)?\.layers\.(\d+)\.mlp\.shared_experts\.)"
    r"(gate|up|down)_proj\.weight$"
)
ASSETS = (
    "tokenizer.json", "tokenizer_config.json", "generation_config.json",
    "special_tokens_map.json", "chat_template.jinja", "processor_config.json",
    "preprocessor_config.json", "tokenizer.model", "vocab.json", "merges.txt",
)
SINGLE_STREAM = "shared_ffn_swiglu_v1"
FOUR_STREAMS = "shared_ffn_four_streams_v2"


def sha256(path: Any) -> Any:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def dimensions(config: Any) -> Any:
    text = config.get("text_config", config)
    if text.get("model_type") != "glm5_next_text":
        raise ValueError("This exporter is qualified only for GLM-5.3-Flash glm5_next_text")
    for name in ("hidden_size", "moe_intermediate_size", "n_shared_experts",
                 "n_routed_experts", "num_hidden_layers"):
        value = text.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"Configuration requires a positive integer {name}")
    if text["n_shared_experts"] != 1:
        raise ValueError("Input must have the contracted one shared expert")
    limit = text.get("swiglu_limit", 10.0)
    if not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0:
        raise ValueError("Invalid SwiGLU limit")
    quant = config.get("quantization") or config.get("quantization_config") or {}
    if quant and str(quant.get("quant_method", "")).lower() != "exl3":
        raise ValueError("Input must be floating weights or routed-only EXL3 with floating shared weights")
    return text, text["moe_intermediate_size"], limit


def read_additions(paths: Any, text: Any, limit: Any) -> Any:
    """Validate both rank artifacts while preserving their full output contract."""
    if len(paths) != 2:
        raise ValueError("Exactly two ordered rank artifacts are required")
    additions = []
    for rank, path in enumerate(paths):
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(artifact, dict) or artifact.get("schema_version") != 1
                or type(artifact.get("rank")) is not int or artifact["rank"] != rank
                or artifact.get("world") != 2 or artifact.get("base_frozen") is not True
                or artifact.get("mechanism") not in (SINGLE_STREAM, FOUR_STREAMS)):
            raise ValueError(f"Rank {rank} artifact contract differs")
        streams = 4 if artifact["mechanism"] == FOUR_STREAMS else 1
        if streams == 4 and text.get("hc_mult", 4) != 4:
            raise ValueError("Four-stream addition requires hc_mult=4")
        if streams == 4:
            study = artifact.get("study_sha256")
            if not isinstance(study, str) or not re.fullmatch(r"[a-f0-9]{64}", study):
                raise ValueError("Four-stream addition requires a sealed study_sha256")
        if artifact.get("swiglu_limit", 10.0) != limit:
            raise ValueError("Training and checkpoint SwiGLU limits differ")
        layers = artifact.get("layers")
        if not isinstance(layers, dict) or not layers:
            raise ValueError("Addition has no layers")
        for layer, site in layers.items():
            if type(layer) is not int or not 0 <= layer < text["num_hidden_layers"]:
                raise ValueError("Addition layer is outside the model")
            if not isinstance(site, dict) or set(site) != {"gate", "up", "down"}:
                raise ValueError("Each site requires only gate/up/down weights")
            gate = site["gate"]
            if not isinstance(gate, torch.Tensor) or gate.ndim != 2:
                raise ValueError("Gate must be a matrix")
            width, hidden = gate.shape
            if width <= 0 or hidden != text["hidden_size"]:
                raise ValueError("Addition hidden dimension differs from config")
            for name in ("gate", "up", "down"):
                tensor = site[name]
                expected = (streams * hidden, width) if name == "down" else (width, hidden)
                if (not isinstance(tensor, torch.Tensor) or tensor.shape != expected
                        or tensor.layout != torch.strided
                        or tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16)
                        or not torch.isfinite(tensor).all()):
                    raise ValueError(f"Invalid finite {name} matrix")
        additions.append(artifact)
    if additions[0]["mechanism"] != additions[1]["mechanism"]:
        raise ValueError("Both ranks must use the same shared output mechanism")
    if (additions[0]["mechanism"] == FOUR_STREAMS
            and additions[0]["study_sha256"] != additions[1]["study_sha256"]):
        raise ValueError("Four-stream ranks name different study hashes")
    if set(additions[0]["layers"]) != set(additions[1]["layers"]):
        raise ValueError("Both ranks must name the same addition layers")
    return additions


def stock_additions(artifacts: Any, hidden: Any) -> Any:
    """A stock shared branch supplies one hidden vector before mHC post mixing."""
    additions = []
    for artifact in artifacts:
        layers = {}
        for layer, site in artifact["layers"].items():
            down = site["down"]
            if artifact["mechanism"] == FOUR_STREAMS:
                first = down[:hidden]
                for stream in range(1, 4):
                    block = down[stream * hidden:(stream + 1) * hidden]
                    if not torch.equal(block, first):
                        raise ValueError(
                            "Four-stream down blocks differ; stock merge requires identical blocks "
                            "because stock mHC broadcasts one hidden vector. "
                            "Export a portable bundle and use a four-stream inference loader."
                        )
                down = first
            layers[layer] = {"gate": site["gate"], "up": site["up"], "down": down}
        additions.append(layers)
    return additions


def load_additions(paths: Any, text: Any, limit: Any) -> Any:
    artifacts = read_additions(paths, text, limit)
    return stock_additions(artifacts, text["hidden_size"])


def checkpoint_catalog(base: Any) -> Any:
    catalog = {}
    shards = {}
    for path in sorted(base.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as reader:
            names = list(reader.keys())
        shards[path.name] = names
        for name in names:
            if name in catalog:
                raise ValueError(f"Duplicate checkpoint tensor {name}")
            catalog[name] = path.name
    if not catalog:
        raise ValueError("Input has no safetensors checkpoint")
    index_path = base / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if index.get("weight_map") != catalog:
            raise ValueError("Checkpoint index and actual tensor files differ")
    return catalog, shards


def shared_sites(catalog: Any, text: Any, additions: Any) -> Any:
    sites: dict[Any, Any] = {}
    for name in catalog:
        match = SHARED.fullmatch(name)
        if match is None:
            continue
        prefix, layer, projection = match.groups()
        layer = int(layer)
        if layer in sites and sites[layer]["prefix"] != prefix:
            raise ValueError("Ambiguous model prefix")
        sites.setdefault(layer, {"prefix": prefix, "names": {}})
        sites[layer]["names"][projection] = name
    kinds = text.get("mlp_layer_types")
    if kinds is None:
        dense = text.get("first_k_dense_replace", 3)
        kinds = ["dense"] * dense + ["sparse"] * (text["num_hidden_layers"] - dense)
    if len(kinds) != text["num_hidden_layers"]:
        raise ValueError("MLP layer types and layer count differ")
    expected = set()
    for layer, kind in enumerate(kinds):
        if kind == "sparse":
            expected.add(layer)
        elif kind != "dense":
            raise ValueError("Unrecognized MLP layer type")
    mtp_count = text.get("num_nextn_predict_layers", 0)
    for layer in range(text["num_hidden_layers"], text["num_hidden_layers"] + mtp_count):
        expected.add(layer)
    if set(sites) != expected:
        raise ValueError("Checkpoint must contain every configured shared expert, including MTP")
    for site in sites.values():
        if set(site["names"]) != {"gate", "up", "down"}:
            raise ValueError("Incomplete shared expert")
    if not set(additions[0]).issubset(expected):
        raise ValueError("Addition targets a dense layer")
    return sites


def merged_matrix(base: Any, name: Any, projection: Any, layer: Any, additions: Any, old_width: Any, new_width: Any, hidden: Any) -> Any:
    axis = 1 if projection == "down" else 0
    expected = (hidden, old_width) if axis == 1 else (old_width, hidden)
    if (base.shape != expected or base.dtype not in (torch.float32, torch.float16, torch.bfloat16)
            or not torch.isfinite(base).all()):
        raise ValueError(f"{name}: base shared matrix differs from configuration or is not finite floating point")
    parts = []
    rounding_max = 0.0
    for rank in range(2):
        original = base.narrow(axis, rank * (old_width // 2), old_width // 2)
        parts.append(original)
        added_width = 0
        if layer in additions[rank]:
            trained = additions[rank][layer][projection]
            added = trained.to(base.dtype)
            if not torch.isfinite(added).all():
                raise ValueError("Addition overflows checkpoint dtype")
            rounding_max = max(rounding_max, float((trained.float() - added.float()).abs().max()))
            parts.append(added)
            added_width = added.shape[axis]
        padding = new_width // 2 - old_width // 2 - added_width
        shape = list(expected)
        shape[axis] = padding
        parts.append(torch.zeros(shape, dtype=base.dtype))
    return torch.cat(parts, dim=axis).contiguous(), rounding_max


def rewrite_shard(source: Any, target: Any, replacements: Any) -> Any:
    """Stream unchanged tensor bytes verbatim, including packed routed experts."""
    with source.open("rb") as reader:
        header_size = struct.unpack("<Q", reader.read(8))[0]
        header = json.loads(reader.read(header_size))
        base_offset = 8 + header_size
        new_header = {}
        if "__metadata__" in header:
            new_header["__metadata__"] = header["__metadata__"]
        payloads = {}
        offset = 0
        for name, info in header.items():
            if name == "__metadata__":
                continue
            new_info = dict(info)
            if name in replacements:
                encoded = save({name: replacements[name]})
                encoded_header_size = struct.unpack("<Q", encoded[:8])[0]
                encoded_header = json.loads(encoded[8:8 + encoded_header_size])
                new_info = encoded_header[name]
                payloads[name] = encoded[8 + encoded_header_size:]
                size = len(payloads[name])
            else:
                size = info["data_offsets"][1] - info["data_offsets"][0]
            new_info["data_offsets"] = [offset, offset + size]
            new_header[name] = new_info
            offset += size
        encoded_header = json.dumps(new_header, separators=(",", ":")).encode()
        encoded_header += b" " * (-len(encoded_header) % 8)
        with target.open("wb") as writer:
            writer.write(struct.pack("<Q", len(encoded_header)))
            writer.write(encoded_header)
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                if name in payloads:
                    writer.write(payloads[name])
                    continue
                first, stop = info["data_offsets"]
                reader.seek(base_offset + first)
                remaining = stop - first
                while remaining:
                    block = reader.read(min(remaining, 8 * 1024 * 1024))
                    if not block:
                        raise ValueError("Truncated source checkpoint")
                    writer.write(block)
                    remaining -= len(block)
    return offset


def export_checkpoint(base: Any, rank_paths: Any, output: Any) -> Any:
    base = Path(base).resolve()
    output = Path(output).resolve()
    if output == base or output.is_relative_to(base) or base.is_relative_to(output):
        raise ValueError("Input and output must be separate checkpoint trees")
    if output.exists():
        raise ValueError("Output already exists; choose a fresh checkpoint directory")
    config = json.loads((base / "config.json").read_text())
    text, old_width, limit = dimensions(config)
    if old_width % 2:
        raise ValueError("Base shared width must split evenly across TP=2")
    artifacts = read_additions(rank_paths, text, limit)
    additions = stock_additions(artifacts, text["hidden_size"])
    catalog, shards = checkpoint_catalog(base)
    shared_sites(catalog, text, additions)
    largest_rank_width = 0
    for rank in range(2):
        for site in additions[rank].values():
            largest_rank_width = max(largest_rank_width, site["gate"].shape[0])
    required_width = old_width + 2 * largest_rank_width
    new_width = math.ceil(required_width / old_width) * old_width
    updated = copy.deepcopy(config)
    updated_text = updated.get("text_config", updated)
    updated_text["n_shared_experts"] = new_width // old_width
    report = {
        "schema_version": 1, "base_config_sha256": sha256(base / "config.json"),
        "addition_mechanism": artifacts[0]["mechanism"],
        "study_sha256": artifacts[0].get("study_sha256"),
        "four_stream_reduction": "identical_down_blocks" if artifacts[0]["mechanism"] == FOUR_STREAMS else None,
        "rank_artifact_sha256": [sha256(rank_paths[0]), sha256(rank_paths[1])],
        "old_shared_width": old_width, "new_shared_width": new_width,
        "config_n_shared_experts": updated_text["n_shared_experts"],
        "shared_mlp_branches_per_layer": 1, "routed_experts": text["n_routed_experts"],
        "added_layers": sorted(additions[0]), "dtype_rounding_max": 0.0, "shards": {},
        "source_quantization": config.get("quantization") or config.get("quantization_config"),
        "fresh_exl3_quantization_performed": False, "stock_hosting_answer_verified": False,
        "artifact_base_binding": "caller-supplied; trainer artifacts carry no base checkpoint identity",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        total_size = 0
        for filename, names in shards.items():
            replacements = {}
            with safe_open(base / filename, framework="pt", device="cpu") as reader:
                for name in names:
                    match = SHARED.fullmatch(name)
                    if match is None:
                        continue
                    _, layer, projection = match.groups()
                    merged, rounding = merged_matrix(reader.get_tensor(name), name, projection,
                                                     int(layer), additions, old_width, new_width,
                                                     text["hidden_size"])
                    replacements[name] = merged
                    report["dtype_rounding_max"] = max(report["dtype_rounding_max"], rounding)
            total_size += rewrite_shard(base / filename, staging / filename, replacements)
            with safe_open(staging / filename, framework="pt", device="cpu") as reader:
                if set(reader.keys()) != set(names):
                    raise ValueError("Export tensor inventory changed")
            report["shards"][filename] = {
                "source_sha256": sha256(base / filename), "output_sha256": sha256(staging / filename),
                "changed_tensors": sorted(replacements),
            }
        for filename in ASSETS:
            if (base / filename).is_file():
                shutil.copy2(base / filename, staging / filename)
        index = {"metadata": {"total_size": total_size}, "weight_map": catalog}
        (staging / "config.json").write_text(json.dumps(updated, indent=2) + "\n")
        (staging / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
        (staging / "export-report.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return report


def main() -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--rank0", type=Path, required=True)
    parser.add_argument("--rank1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = export_checkpoint(args.base, [args.rank0, args.rank1], args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
