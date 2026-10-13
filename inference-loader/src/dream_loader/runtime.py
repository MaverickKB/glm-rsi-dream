"""Fixed shared-expert inference, installed before stock CUDA graph capture."""

import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from .artifact import FOUR_STREAMS, digest, geometry, load_pair

KEY = "glm_dream_loader"
ENV = "TF_GLM_DREAM_MANIFEST"


def load_local(engine, model_dir):
    w = engine.w
    if w.world != 2 or w.rank not in (0, 1) or w.cfg.quant != "exl3":
        raise ValueError("Shared addition requires stock EXL3 TP=2")
    path = os.environ.get(ENV)
    if not path:
        raise ValueError(ENV + " is required on both ranks")
    manifest, pair = load_pair(path, w.device, w.rank)
    binding = manifest["binding"]
    raw_config = (Path(model_dir) / "config.json").read_bytes()
    if digest(raw_config) != binding["model_config_sha256"]:
        raise ValueError("Addition model config identity mismatch")
    shape = geometry(json.loads(raw_config))
    if shape != binding["geometry"] or w.cfg.hidden != shape["hidden_size"] or w.cfg.limit != 10:
        raise ValueError("Loaded model geometry or SwiGLU clamp differs")
    if binding["mechanism"] == FOUR_STREAMS and w.cfg.streams != 4:
        raise ValueError("Four-stream addition requires four native HC streams")
    for index in pair[w.rank]:
        if index >= len(w.layers) or w.layers[index].moe.shared is None:
            raise ValueError("Addition target has no EXL3 shared expert")
    if KEY in w.meta:
        raise ValueError("Shared addition was already attached")
    return {"layers": pair[w.rank], "pair_sha256": manifest["pair_sha256"],
            "mechanism": binding["mechanism"]}


def attach(engine, model_dir):
    state = None
    failure = None
    try:
        state = load_local(engine, model_dir)
    except Exception as exc:
        failure = exc
    # Use the stock startup exchange even when local validation fails. Both
    # ranks then reject the startup instead of leaving one waiting in inference.
    pair_hash = "0" * 64 if state is None else state["pair_sha256"]
    mine = [int(state is not None)] + list(bytes.fromhex(pair_hash))
    rows = engine._gather_ints(mine)
    valid = len(rows) == 2 and rows[0] == mine and rows[1] == mine and mine[0] == 1
    if not valid:
        raise ValueError("Serving ranks reject or disagree on the paired addition") from failure
    w = engine.w
    w.meta[KEY] = state
    print(json.dumps({"loader": KEY, "rank": w.rank, "world": 2,
                      "pair_sha256": pair_hash, "layers": list(state["layers"]),
                      "mechanism": state["mechanism"],
                      "attention_pockets": False, "learning": False}), flush=True)


@torch.inference_mode()
def shared(layer, w, b, lo, hi):
    state = w.meta.get(KEY)
    if state is None:
        raise RuntimeError("Shared addition was not attached before inference")
    site = state["layers"].get(layer.index)
    if site is None or hi <= lo:
        return
    inputs = b.normed[lo:hi].float()
    gate = F.linear(inputs, site["gate"]).clamp(max=w.cfg.limit)
    up = F.linear(inputs, site["up"]).clamp(-w.cfg.limit, w.cfg.limit)
    activation = F.silu(gate) * up
    correction = F.linear(activation, site["down"])
    if four_streams(w):
        b._dream_delta[lo:hi].add_(correction.reshape(hi - lo, 4, w.cfg.hidden))
    else:
        b.sy[lo:hi].add_(correction)


def four_streams(w):
    return w.meta[KEY]["mechanism"] == FOUR_STREAMS


@torch.inference_mode()
def before_ffn(layer, w, b, count):
    b._dream_delta = None
    b._dream_before = None
    if not four_streams(w) or layer.index not in w.meta[KEY]["layers"]:
        return
    b._dream_before = b.x[:count].reshape(count, 4, w.cfg.hidden).clone()
    b._dream_delta = torch.zeros(count, 4, w.cfg.hidden, dtype=torch.float32, device=w.device)


@torch.inference_mode()
def after_ffn(layer, w, b, count, gathered):
    if b._dream_delta is None:
        return
    partial = b._dream_delta.contiguous()
    all_delta = torch.empty((w.world, *partial.shape), dtype=partial.dtype, device=w.device)
    w.comm.all_gather(partial.reshape(-1), all_delta.reshape(-1))
    correction = all_delta[0]
    base = gathered[0, :count].float()
    for rank in range(1, w.world):
        correction = correction + all_delta[rank]
        base = base + gathered[rank, :count].float()
    streams = b._dream_before.float()
    combine = b.comb[:count].reshape(count, 4, 4)
    post = b.post[:count].reshape(count, 4)
    output = []
    for index in range(4):
        mixed = streams[:, 0] * combine[:, 0, index, None]
        for source in range(1, 4):
            mixed = mixed + streams[:, source] * combine[:, source, index, None]
        branch = (base + correction[:, index]).to(b.x.dtype).float()
        output.append((mixed + post[:, index, None] * branch).to(b.x.dtype))
    b.x[:count].copy_(torch.stack(output, dim=1).reshape_as(b.x[:count]))
    b._dream_delta = None
    b._dream_before = None
