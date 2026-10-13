"""Validate the DREAM rank tensors and their paired serving identity."""

import hashlib
import io
import json
from pathlib import Path

import torch

MECHANISM = "shared_ffn_swiglu_v1"
FOUR_STREAMS = "shared_ffn_four_streams_v2"
MECHANISMS = (MECHANISM, FOUR_STREAMS)
MODEL = "GLM-5.3-Flash-EXL3"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return digest(raw)


def geometry(config):
    text = config.get("text_config", config)
    fields = ("hidden_size", "moe_intermediate_size", "n_shared_experts",
              "n_routed_experts", "num_hidden_layers", "first_k_dense_replace")
    result = {}
    for field in fields:
        value = text.get(field)
        if type(value) is not int or value < 1:
            raise ValueError("Missing or invalid model geometry: " + field)
        result[field] = value
    if (result["hidden_size"] != 4096 or result["moe_intermediate_size"] != 2048
            or result["n_shared_experts"] != 1 or result["n_routed_experts"] != 288
            or result["num_hidden_layers"] != 45 or result["first_k_dense_replace"] != 3):
        raise ValueError("Only the observed GLM-5.3-Flash geometry is supported")
    return result


def read_rank(raw, rank, hidden, device="cpu"):
    # Deserialize the verified bytes, rather than reopening a mutable path.
    value = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
    if (not isinstance(value, dict) or value.get("schema_version") != 1
            or value.get("mechanism") not in MECHANISMS or type(value.get("rank")) is not int
            or value["rank"] != rank or value.get("world") != 2
            or value.get("base_frozen") is not True):
        raise ValueError("Addition schema, mechanism or rank ownership mismatch")
    layers = value.get("layers")
    streams = 4 if value["mechanism"] == FOUR_STREAMS else 1
    if not isinstance(layers, dict) or not layers:
        raise ValueError("Addition needs at least one middle shared-expert layer")
    result = {}
    for index, site in layers.items():
        if type(index) is not int or not 3 <= index <= 39:
            raise ValueError("Addition contains a non-middle MoE layer")
        if not isinstance(site, dict) or set(site) != {"gate", "up", "down"}:
            raise ValueError("Each shared site needs gate, up and down only")
        for tensor in site.values():
            if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                    or tensor.dtype != torch.float32 or tensor.ndim != 2
                    or not torch.isfinite(tensor).all()):
                raise ValueError("Addition tensors must be finite, dense FP32 matrices")
        width, actual_hidden = site["gate"].shape
        if (width < 1 or actual_hidden != hidden or site["up"].shape != (width, hidden)
                or site["down"].shape != (hidden * streams, width)):
            raise ValueError("Addition gate/up/down geometry mismatch")
        result[index] = {}
        for name, tensor in site.items():
            result[index][name] = tensor.detach().to(device).contiguous()
    return value["mechanism"], result


def load_pair(path, device="cpu", rank=None):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    binding = manifest["binding"]
    if (manifest.get("schema_version") != 1 or binding.get("mechanism") not in MECHANISMS
            or binding.get("model") != MODEL or binding.get("world") != 2
            or manifest.get("pair_sha256") != identity(binding)):
        raise ValueError("Paired addition identity mismatch")
    shape = geometry(binding["geometry"])
    selected = binding["ranks"]
    if not isinstance(selected, list) or len(selected) != 2:
        raise ValueError("Exactly two rank artifacts are required")
    if rank is not None and (type(rank) is not int or rank not in (0, 1)):
        raise ValueError("Only TP ranks 0 and 1 are supported")
    pair = []
    for index, item in enumerate(selected):
        file = (path.parent / item["path"]).resolve()
        if item.get("rank") != index or not file.is_relative_to(path.parent):
            raise ValueError("Rank order or artifact path boundary mismatch")
        raw = file.read_bytes()
        if digest(raw) != item["sha256"]:
            raise ValueError("Rank artifact hash mismatch")
        target = device if rank is None or index == rank else "cpu"
        mechanism, layers = read_rank(raw, index, shape["hidden_size"], target)
        if mechanism != binding["mechanism"]:
            raise ValueError("Rank artifact mechanism differs from paired binding")
        pair.append(layers)
    if set(pair[0]) != set(pair[1]):
        raise ValueError("Both ranks must carry the same shared-expert layers")
    for index in pair[0]:
        if pair[0][index]["gate"].shape != pair[1][index]["gate"].shape:
            raise ValueError("Paired addition widths differ")
    return manifest, pair
