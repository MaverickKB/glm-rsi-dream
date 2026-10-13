"""Native FFN application and aligned captures, with no factual text routing."""

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

KEY = "tensorfold_dream.eval"


def records(w: Any, b: Any) -> Any:
    from tensorfold_dream.eval_native import spec_for

    for state, offset, count in getattr(b, "_eval_pieces", ()):
        record = spec_for(w, state)
        yield record, state, offset, count


def artifact(w: Any, selected: Any) -> Any:
    if selected is None:
        return None
    root = w.meta[KEY]["root"].resolve()
    path = (root / selected["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("FFN artifact escapes paired manifest")
    cache = w.meta.setdefault("dream_neurons", {})
    expected = selected["sha256"]
    if cache.get("sha256") != expected:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("FFN artifact hash mismatch")
        value = torch.load(path, weights_only=True, map_location=w.device)
        if (value.get("mechanism") not in ('shared_ffn_swiglu_v1', 'shared_ffn_four_streams_v2')
                or value.get("rank") != w.rank
                or value.get("world") != 2 or value.get("base_frozen") is not True):
            raise ValueError("FFN mechanism or rank ownership mismatch")
        for index, site in value["layers"].items():
            if not isinstance(index, int) or not 3 <= index <= 39:
                raise ValueError("FFN artifact contains a non-middle layer")
            width, hidden = site["gate"].shape
            streams = 4 if value['mechanism'] == 'shared_ffn_four_streams_v2' else 1
            if hidden != w.cfg.hidden or site["down"].shape != (hidden * streams, width):
                raise ValueError("FFN native shape mismatch")
            if site["up"].shape != (width, hidden):
                raise ValueError("FFN up shape mismatch")
            for tensor in site.values():
                if not torch.isfinite(tensor).all():
                    raise ValueError("Nonfinite FFN artifact")
        cache.update(sha256=expected, artifact=value)
    return cache["artifact"]


def selected_records(w: Any, b: Any) -> Any:
    state = w.meta.get(KEY)
    if state is None:
        return
    current_path = state["root"] / "neurons-current.json"
    current = None
    if current_path.exists():
        current = json.loads(current_path.read_text())["by_rank"][w.rank]
    for record, owned, offset, count in records(w, b):
        policy = {} if record is None else record["policy"]
        selected = policy.get("neurons", current)
        if policy.get("neurons_off"):
            selected = None
        value = artifact(w, selected)
        yield record, offset, count, selected, value


def shared(layer: Any, w: Any, b: Any, lo: Any, hi: Any) -> Any:
    for record, offset, count, selected, value in selected_records(w, b):
        if value is None or layer.index not in value["layers"]:
            continue
        first = max(lo, offset)
        stop = min(hi, offset + count)
        if stop <= first:
            continue
        site = value["layers"][layer.index]
        inputs = b.normed[first:stop].float()
        gate = F.linear(inputs, site["gate"]).clamp(max=w.cfg.limit)
        up = F.linear(inputs, site["up"]).clamp(-w.cfg.limit, w.cfg.limit)
        activation = F.silu(gate) * up
        correction = F.linear(activation, site["down"])
        if value['mechanism'] == 'shared_ffn_four_streams_v2':
            b._dream_delta[first:stop].add_(correction.reshape(stop - first, 4, w.cfg.hidden))
        else:
            b.sy[first:stop].add_(correction)
        if record is not None:
            record["neuron_sha256"] = selected["sha256"]
            record["neuron_rows"] = record.get("neuron_rows", 0) + stop - first


def capture_rows(record: Any, state: Any, offset: Any, count: Any) -> Any:
    chosen = []
    positions = []
    for position in record["policy"]["neuron_capture"]["rows"]:
        relative = position - state.pos
        if 0 <= relative < count:
            chosen.append(offset + relative)
            positions.append(position)
    return chosen, positions


def before_ffn(layer: Any, w: Any, b: Any, count: Any) -> Any:
    b._dream_delta = None
    b._dream_before = None
    for _, _, _, _, value in selected_records(w, b):
        if (value is not None and value['mechanism'] == 'shared_ffn_four_streams_v2'
                and layer.index in value['layers']):
            b._dream_before = b.x[:count].reshape(count, 4, w.cfg.hidden).clone()
            b._dream_delta = torch.zeros(count, 4, w.cfg.hidden, dtype=torch.float32, device=w.device)
            break
    for record, state, offset, size in records(w, b):
        if record is None:
            continue
        spec = record["policy"].get("neuron_capture")
        if spec is None or layer.index not in spec["layers"]:
            continue
        if not b.prefill and not spec.get('decode_trace'):
            continue
        if b.split is not None and b.split.active:
            raise ValueError("FFN capture requires the declared unsplit paired candidate")
        chosen, positions = capture_rows(record, state, offset, size)
        if not chosen:
            continue
        completed = record.get('neuron_chunks', {}).get(layer.index, [])
        captured_rows = 0
        for chunk in completed:
            captured_rows += len(chunk['positions'])
        if captured_rows + len(chosen) > 512 or len(spec["layers"]) > 12:
            raise ValueError("FFN capture exceeds bounded geometry")
        record.setdefault("neuron_pending", {})[layer.index] = {
            "positions": positions, "input": b.normed[chosen].detach().cpu().clone(),
            "post": b.post[chosen].detach().cpu().clone()}
        if spec.get('answer_trace'):
            pending = record['neuron_pending'][layer.index]
            pending['before'] = b.x[chosen].detach().cpu().reshape(len(chosen), 4, w.cfg.hidden).clone()
            pending['combine'] = b.comb[chosen].detach().cpu().reshape(len(chosen), 4, 4).clone()


def mix_branch(before: Any, output: Any, branch: Any, post: Any, combine: Any) -> Any:
    from tensorfold.families.glm5_next.cuda import glue

    glue.hc_post(before, output, branch.unsqueeze(0).contiguous(), post, combine)


def after_ffn(layer: Any, w: Any, b: Any, count: Any, gathered: Any = None) -> Any:
    if b._dream_delta is not None:
        partial = b._dream_delta.contiguous()
        all_delta = torch.empty((w.world, *partial.shape), dtype=partial.dtype, device=w.device)
        w.comm.all_gather(partial.reshape(-1), all_delta.reshape(-1))
        correction = all_delta[0]
        base = gathered[0, :count].float()
        for rank in range(1, w.world):
            correction = correction + all_delta[rank]
            base = base + gathered[rank, :count].float()
        before = b._dream_before.reshape_as(b.x[:count])
        output = torch.empty_like(b.x[:count])
        for index in range(4):
            branch = base + correction[:, index]
            mix_branch(before, output, branch, b.post[:count], b.comb[:count])
            b.x[:count].reshape(count, 4, w.cfg.hidden)[:, index].copy_(
                output.reshape(count, 4, w.cfg.hidden)[:, index])
        b._dream_delta = None
        b._dream_before = None
    for record, state, offset, size in records(w, b):
        if record is None:
            continue
        pending = record.get("neuron_pending", {}).pop(layer.index, None)
        if pending is None:
            continue
        chosen, positions = capture_rows(record, state, offset, size)
        if positions != pending["positions"]:
            raise ValueError("FFN capture row identity changed")
        pending["residual"] = b.x[chosen].detach().cpu().reshape(len(chosen), 4, w.cfg.hidden).clone()
        if 'before' in pending:
            if gathered is None or gathered.shape[0] != w.world:
                raise ValueError('Answer trace requires the actual paired FFN reduction')
            reduced = gathered[0, chosen].float()
            for rank in range(1, w.world):
                reduced = reduced + gathered[rank, chosen].float()
            pending['base_output'] = reduced.detach().cpu().clone()
        record.setdefault("neuron_chunks", {}).setdefault(layer.index, []).append(pending)


def finished(w: Any, record: Any) -> Any:
    receipt = {"neuron_sha256": record.get("neuron_sha256"),
               "neuron_rows": record.get("neuron_rows", 0)}
    chunks = record.pop("neuron_chunks", None)
    if chunks:
        packet = {"schema_version": 1, "rank": w.rank, "world": w.world,
                  "request_id": record["spec"]["request_id"],
                  "attention_memory": bool(record.get("memory_activated")), "layers": chunks}
        path = Path("/capture") / ("neurons-" + packet["request_id"] + "-rank" + str(w.rank) + ".pt")
        torch.save(packet, path)
        receipt["neuron_capture_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return receipt
