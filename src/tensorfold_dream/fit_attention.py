# SPDX-License-Identifier: Apache-2.0
"""Real gradients of a frozen native MLA readout reference, not full-model KL."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from tensorfold.families.glm5_next.cuda import kv8

from glm_rsi_dream.attention import attention
from glm_rsi_dream.manifest import MANIFEST


def main() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--slots',type=int,default=MANIFEST['memory']['slots'])
    parser.add_argument('--output',default='compiled')
    args = parser.parse_args()
    slot_count = args.slots
    if slot_count != MANIFEST["memory"]["slots"]:
        raise ValueError("This release fits 96 aligned memory rows")
    torch.set_num_threads(4)
    torch.manual_seed(MANIFEST['memory']['seed'])
    rank = args.rank
    root = Path('/opt/pocket/pair')
    captures = Path('/capture')
    index = json.loads((root / 'train-index.json').read_text())
    full = torch.load(captures / ('fullstate-rank' + str(rank) + '.pt'), map_location='cpu', weights_only=True)
    n = full['length']
    layers = MANIFEST['memory']['layers']
    output_rows = list(full['rows'])
    summaries = []
    for layer_number, layer in enumerate(layers):
        teacher = kv8.dequantize(full['rows'][layer_number].to('cuda')).detach()
        initial_ids = torch.linspace(0, n - 1, slot_count, device='cuda').round().long()
        memory = torch.nn.Parameter(teacher.index_select(0, initial_ids).clone())
        optimizer = torch.optim.Adam([memory], lr=MANIFEST['memory']['learning_rate'])
        dataset = []
        reference_errors = []
        for entry in index:
            packet = torch.load(captures / (entry['request_id'] + '-rank' + str(rank) + '.pt'), map_location='cpu', weights_only=True)
            for part in packet.get(layer, []):
                begin = max(0, entry['answer_start'] - part['position'])
                queries = part['queries'][begin:].to('cuda')
                if not queries.shape[0]:
                    continue
                rows = kv8.dequantize(part['cache'].to('cuda'))
                history = rows[n:].detach()
                positions = torch.arange(part['position'] + begin - n, part['position'] + part['queries'].shape[0] - n, device='cuda')
                with torch.no_grad():
                    target = attention(queries, teacher, history, positions)
                if 'native_output' in part:
                    observed = part['native_output'][begin:].to('cuda')
                    reference_errors.append(float((target - observed).norm() / observed.norm().clamp(min=1e-8)))
                queries = queries.detach()
                dataset.append((queries, history, positions, target))
        if not reference_errors or max(reference_errors)>0.01:
            raise RuntimeError('Attention reference differs from native output')
        if not dataset:
            raise RuntimeError('No training trajectory at layer ' + str(layer))
        queries, history, positions, target = dataset[0]
        loss = (attention(queries, memory, history, positions) - target).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite attention loss")
        loss.backward()
        gradient = memory.grad
        if gradient is None or not torch.isfinite(gradient).all() or gradient.norm() == 0:
            raise RuntimeError("Invalid or zero attention gradient")
        direction = torch.randn_like(memory)
        direction /= direction.norm()
        derivative = float((gradient * direction).sum())
        epsilon = 0.02
        with torch.no_grad():
            plus = (attention(queries.double(), (memory + epsilon * direction).double(), history.double(), positions) - target.double()).square().mean()
            minus = (attention(queries.double(), (memory - epsilon * direction).double(), history.double(), positions) - target.double()).square().mean()
            numeric = float((plus - minus) / (2 * epsilon))
        relative = abs(derivative - numeric) / max(abs(derivative), abs(numeric), 1e-8)
        if relative > 0.08:
            raise RuntimeError('Attention gradient directional check failed')
        losses = []
        for step in range(MANIFEST['memory']['steps']):
            queries, history, positions, target = dataset[step % len(dataset)]
            optimizer.zero_grad(set_to_none=True)
            prediction = attention(queries, memory, history, positions)
            loss = (prediction - target).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite fitting loss")
            loss.backward()
            gradient = memory.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError("Nonfinite fitting gradient")
            optimizer.step()
            if step % 50 == 0:
                losses.append({'step': step, 'loss': float(loss.detach())})
        with torch.no_grad():
            encoded = kv8.quantize_rows(memory.detach().to(torch.bfloat16))
            first = dataset[0]
            post = (attention(first[0], kv8.dequantize(encoded), first[1], first[2]) - first[3]).square().mean()
        if not torch.isfinite(kv8.dequantize(encoded)).all() or not torch.isfinite(post):
            raise RuntimeError("Quantized memory contains nonfinite values")
        output_rows[layer_number] = encoded.cpu()
        summaries.append({'layer': layer, 'training_trajectories': len(dataset), 'derivative': derivative,
                          'finite_difference': numeric, 'relative_error': relative, 'losses': losses,
                          'native_reference_errors': reference_errors, 'post_fp8_loss_first_train': float(post),
                          'parameter_change_norm': float((memory.detach() - teacher.index_select(0, initial_ids)).norm())})
        print('rank', rank, 'layer', layer, 'gradient_error', relative, 'loss', losses[-1]['loss'], flush=True)
        del dataset, teacher, memory, optimizer, queries, history, positions, target, prediction, loss, first
        torch.cuda.empty_cache()
    for i in range(len(layers), len(output_rows)):
        output_rows[i] = output_rows[i][:slot_count // 4 + 1].clone()
    tail = []
    for index_slots, values in full['tail']:
        tail.append((torch.empty(0, dtype=index_slots.dtype), values[:, :0].clone()))
    artifact = {'schema_version': 3, 'mechanism': 'attention_state_v1', 'rank': rank, 'length': slot_count,
                'rows': output_rows, 'tail': tail, 'training_only': True, 'dense_only': True}
    path = captures / (args.output+'-rank' + str(rank) + '.pt')
    torch.save(artifact, path)
    report = {'rank': rank, 'objective': 'native attention-output MSE; frozen teacher queries, not end-to-end vocabulary KL',
              'layers': summaries, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'artifact_bytes': path.stat().st_size, 'cuda_peak_allocated': torch.cuda.max_memory_allocated()}
    (captures / (args.output+'-report-rank' + str(rank) + '.json')).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
