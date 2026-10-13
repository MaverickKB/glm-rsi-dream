"""Exact local-gradient fitting of new FFN neurons from native state pairs."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .scope import verify_scope


class AddedSharedNeurons(nn.Module):
    """Added gate/up/down neurons in one frozen-base shared-expert pocket."""

    def __init__(self, hidden: Any, width: Any, streams: Any = 1) -> None:
        super().__init__()
        if streams not in (1, 4):
            raise ValueError('Shared output requires one or four residual streams')
        self.streams = streams
        self.gate = nn.Linear(hidden, width, bias=False)
        self.up = nn.Linear(hidden, width, bias=False)
        self.down = nn.Linear(width, hidden * streams, bias=False)
        nn.init.zeros_(self.down.weight)

    def forward(self, inputs: Any) -> Any:
        gate = self.gate(inputs).clamp(max=10)
        up = self.up(inputs).clamp(-10, 10)
        activations = F.silu(gate) * up
        output = self.down(activations)
        if self.streams == 4:
            output = output.reshape(*inputs.shape[:-1], 4, inputs.shape[-1])
        return output


def projected_correction(student: Any, teacher: Any, post: Any, world: Any = 2) -> Any:
    """Least-squares FFN correction through the four-stream HC post weights.

    Each rank adds one half of the full correction before native all-gather.
    The residual component outside the post direction remains measurable.
    """
    error = teacher.float() - student.float()
    weights = post.float().unsqueeze(-1)
    denominator = weights.square().sum(1).clamp_min(1e-8)
    full = (weights * error).sum(1) / denominator
    unexplained = error - weights * full.unsqueeze(1)
    return full / world, unexplained


def fit_site(inputs: Any, targets: Any, positive: Any, width: Any = 256, steps: Any = 1000, rate: Any = 0.003, seed: Any = 17,
             first_question: Any = None) -> Any:
    """Fit only captured training rows. No development or holdout reaches here."""
    if inputs.ndim != 2 or targets.shape != inputs.shape:
        raise ValueError("FFN inputs and correction targets must share [rows, hidden]")
    if positive.shape != (inputs.shape[0],) or positive.dtype != torch.bool:
        raise ValueError("Every row requires an explicit positive/negative training label")
    if not positive.any() or positive.all():
        raise ValueError("Fitting requires positive and preservation training rows")
    if not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
        raise ValueError("Nonfinite native capture")
    if first_question is not None:
        if (first_question.dtype != torch.bool or first_question.shape != positive.shape
                or not first_question.any() or (first_question & ~positive).any()):
            raise ValueError('Bare-question rows must be a nonempty subset of positive training rows')
    torch.manual_seed(seed)
    model = AddedSharedNeurons(inputs.shape[1], width).to(inputs.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    scale = targets[positive].square().mean().sqrt().clamp_min(1e-4)
    # Added neurons use the existing shared expert's clamped SwiGLU form.
    positive_rows = inputs[positive]
    if first_question is not None:
        positive_rows = inputs[first_question]
    with torch.no_grad():
        for index in range(width):
            position = index % len(positive_rows)
            if first_question is not None and width < len(positive_rows):
                position = index * len(positive_rows) // width
            key = positive_rows[position]
            model.gate.weight[index].copy_(key / key.square().sum().clamp_min(1e-8))
            model.up.weight[index].copy_(key / key.square().sum().clamp_min(1e-8))
        if first_question is not None:
            # Unique directions let the full widened expert use its capacity.
            scale_key = positive_rows.norm(dim=1).mean().clamp_min(1e-8)
            noise = 0.01 / (scale_key * inputs.shape[1] ** .5)
            model.gate.weight.add_(torch.randn_like(model.gate.weight) * noise)
            model.up.weight.add_(torch.randn_like(model.up.weight) * noise)
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        recall_loss = ((output[positive] - targets[positive]) / scale).square().mean()
        if first_question is not None:
            first_loss = ((output[first_question] - targets[first_question]) / scale).square().mean()
            continuation = positive & ~first_question
            if continuation.any():
                remaining = ((output[continuation] - targets[continuation]) / scale).square().mean()
                recall_loss = .5 * first_loss + .5 * remaining
            else:
                recall_loss = first_loss
        preserve_loss = (output[~positive] / scale).square().mean()
        magnitude_loss = (output[positive] / scale).square().mean()
        loss = recall_loss + 4 * preserve_loss + 0.0001 * magnitude_loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite FFN objective")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise ValueError("Missing or nonfinite real FFN gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            history.append({"step": step + 1, "loss": float(loss.detach()),
                            "recall_mse": float(recall_loss.detach()),
                            "preserve_mse": float(preserve_loss.detach())})
    with torch.no_grad():
        output = model(inputs)
        negative_max = float(output[~positive].abs().max())
        relative = float((output[positive] - targets[positive]).norm() /
                         targets[positive].norm().clamp_min(1e-8))
    site = {"gate": model.gate.weight.detach().cpu(),
            "up": model.up.weight.detach().cpu(),
            "down": model.down.weight.detach().cpu()}
    report = {"rows": len(inputs), "positive_rows": int(positive.sum()),
              "negative_rows": int((~positive).sum()), "negative_max_correction": negative_max,
              "relative_correction_error": relative, "history": history,
              "objective": "local projected residual MSE and negative preservation",
              "gradient": "exact PyTorch autograd through added shared neurons, base frozen"}
    if first_question is not None:
        report['first_question_rows'] = int(first_question.sum())
        report['objective'] = 'equal bare-question and continuation MSE, negative preservation'
    return site, report


def partition_shared(site: Any, rank: Any, world: Any = 2) -> Any:
    """Partition one widened shared expert into its native TP neuron shards."""
    width = site['gate'].shape[0]
    if world != 2 or rank not in (0, 1) or width % world:
        raise ValueError('Shared neuron width requires a complete TP=2 partition')
    if site['up'].shape != site['gate'].shape or site['down'].shape[1] != width:
        raise ValueError('Shared gate/up/down geometry differs')
    local = width // world
    start = rank * local
    stop = start + local
    return {'gate': site['gate'][start:stop].clone(),
            'up': site['up'][start:stop].clone(),
            'down': site['down'][:, start:stop].clone()}


def fit_packets(path: Any, output: Any, rank: Any, width: Any = 256, steps: Any = 1000, device: Any = "cuda", progress: Any = None) -> Any:
    if progress is None:
        progress = sys.stdout
    packet = torch.load(path, weights_only=True, map_location=device)
    if packet.get("schema_version") != 1 or packet.get("role") != "train":
        raise ValueError("Only sealed training packets may enter fitting")
    if packet.get("rank") != rank or packet.get("world") != 2:
        raise ValueError("Training packet rank/world differs from the requested artifact")
    root = Path(path).parent
    study_sha, _ = verify_scope(root)
    if packet.get('study_sha256') != study_sha:
        raise ValueError('Training packet differs from the frozen study')
    index_sha = hashlib.sha256((root / 'capture-index.json').read_bytes()).hexdigest()
    if packet.get('index_sha256') != index_sha:
        raise ValueError('Training packet differs from its verified capture index')
    artifact = {"schema_version": 1, "mechanism": "shared_ffn_swiglu_v1",
                "rank": rank, "world": 2, "base_frozen": True,
                "study_sha256": study_sha, "index_sha256": index_sha, "layers": {}}
    reports = {}
    for layer, data in packet["layers"].items():
        if not isinstance(layer, int) or not 3 <= layer <= 39:
            raise ValueError("Neuron consolidation requires middle MoE layers")
        targets, unexplained = projected_correction(data["student"], data["teacher"], data["post"])
        targets[~data["positive"]] = 0
        site, report = fit_site(data["input"].float(), targets, data["positive"], width, steps)
        report["unexpressible_relative_error"] = float(unexplained.norm() /
            (data["teacher"].float() - data["student"].float()).norm().clamp_min(1e-8))
        artifact["layers"][layer] = site
        reports[str(layer)] = report
        print(json.dumps({"layer": layer, **report}), file=progress, flush=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    report = {"rank": rank, "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "training_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              "layers": reports, "native_recall_verified": False}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    fit_packets(args.training, args.output, args.rank, args.width, args.steps, args.device)


if __name__ == "__main__":
    main()
