"""Final-answer fitting of one four-stream shared pocket through a frozen suffix."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .fit import AddedSharedNeurons, partition_shared
from .reference import FrozenSuffix, hyper_post
from .scope import verify_examples, verify_scope


def load_rows(root: Any, example: Any, layer: Any, device: Any = 'cuda') -> Any:
    """Admit both native ranks and every captured sequence position."""
    fields = ('input', 'before', 'base_output', 'post', 'combine', 'residual')
    if len(example['captures']) != 2:
        raise ValueError('Answer fitting requires both native ranks')
    result: Any = None
    for rank, entry in enumerate(example['captures']):
        path = (root / entry['path']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Answer capture escaped its training root')
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Answer capture differs from paired receipt')
        packet = torch.load(path, weights_only=True, map_location='cpu')
        if (packet['schema_version'] != 1 or packet['rank'] != rank or packet['world'] != 2
                or packet['attention_memory']):
            raise ValueError('Answer capture rank or attention-OFF state differs')
        positions = []
        buffers: dict[Any, Any] = {}
        for field in fields:
            buffers[field] = []
        for chunk in packet['layers'][layer]:
            positions.extend(chunk['positions'])
            for field in fields:
                buffers[field].append(chunk[field])
        if positions != list(range(len(example['token_ids']))):
            raise ValueError('Answer training requires a fresh full token sequence')
        local = {}
        for field in fields:
            local[field] = torch.cat(buffers[field])
            if not torch.isfinite(local[field]).all():
                raise ValueError('Answer capture contains nonfinite tensors')
        rows, hidden = local['input'].shape
        shapes = {'before': (rows, 4, hidden), 'residual': (rows, 4, hidden),
                  'base_output': (rows, hidden), 'post': (rows, 4), 'combine': (rows, 4, 4)}
        for field, shape in shapes.items():
            if local[field].shape != shape:
                raise ValueError('Answer capture geometry differs: ' + field)
        if rank == 0:
            result = local
        else:
            for field in fields:
                if not torch.equal(result[field], local[field]):
                    raise ValueError('Paired answer states disagree: ' + field)
    for name, tensor in result.items():
        result[name] = tensor.to(device)
    return result


def interleave_answers(examples: Any) -> Any:
    """Spread unequal fact groups across the complete exposure pass."""
    groups: dict[Any, Any] = {}
    for example in examples:
        name = example['id'].split('-train-')[0]
        groups.setdefault(name, []).append(example)
    cursors = dict.fromkeys(groups, 0)
    ordered: list[Any] = []
    while len(ordered) < len(examples):
        selected = None
        earliest = None
        for name, values in groups.items():
            position = cursors[name]
            if position == len(values):
                continue
            progress = (position + 0.5) / len(values)
            if earliest is None or progress < earliest:
                selected = name
                earliest = progress
        position = cursors[selected]
        ordered.append(groups[selected][position])
        cursors[selected] += 1
    return ordered


def final_answer_loss(model: Any, data: Any, suffix: Any, example: Any, layer: Any) -> Any:
    losses, added = final_answer_batch(model, [data], suffix, [example], layer)
    return losses, added[0]


def final_answer_batch(model: Any, rows: Any, suffix: Any, examples: Any, layer: Any) -> Any:
    """Keep each answer's attention independent while sharing frozen weight reads."""
    streams = []
    additions = []
    for data in rows:
        added = model(data['input'].float())
        streams.append(hyper_post(data['before'], data['base_output'][:, None] + added,
                                  data['post'], data['combine']))
        additions.append(added)
    if len(streams) == 1:
        predictions = [suffix(streams[0], layer + 1)]
    else:
        predictions = suffix.batch(streams, layer + 1)
    losses = []
    for example, logits in zip(examples, predictions, strict=True):
        start = example['answer_start']
        target = torch.tensor(example['target_ids'], device=logits.device)
        tokens = F.cross_entropy(logits[start:start + len(target)], target, reduction='none')
        losses.append(tokens.mean())
    return torch.stack(losses).mean(), additions


def replay_basis(root: Any, examples: Any, layer: Any, device: Any) -> Any:
    """Find the input span to which the added frontend must remain insensitive."""
    rows = []
    for example in examples:
        data = load_rows(root, example, layer, device='cpu')
        rows.append(data['input'].float())
    inputs = torch.unique(torch.cat(rows), dim=0)
    _, singular, right = torch.linalg.svd(inputs, full_matrices=False)
    rank = int((singular > singular[0] * 1e-6).sum())
    if rank == inputs.shape[1]:
        raise ValueError('Ordinary replay leaves no trainable input directions')
    return right[:rank].T.contiguous().to(device)


def remove_replay_component(values: Any, basis: Any) -> Any:
    """Project frontend weights or gradients away from admitted replay inputs."""
    coefficients = values @ basis
    values.sub_(coefficients @ basis.T)


def presentation_count(history: Any) -> Any:
    """Continue the example cursor even when a resumed batch size changes."""
    count = 0
    for row in history:
        count += len(row['example_ids'])
    return count


def load_initial_shared(root: Any, model: Any) -> Any:
    """Load sealed added weights into a new study with a fresh optimizer."""
    root = Path(root).resolve()
    manifest_path = root / 'initialization.json'
    manifest = json.loads(manifest_path.read_text())
    freeze = json.loads((root / 'study-freeze.json').read_text())
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if freeze['files'].get('initialization.json') != manifest_sha:
        raise ValueError('Initialization manifest differs from the sealed study')
    if manifest['role'] != 'added shared weights initialization':
        raise ValueError('Initialization must contain only the added shared weights')
    path = (root / manifest['path']).resolve()
    if not path.is_relative_to(root) or 'initialization.json' not in freeze['files']:
        raise ValueError('Initialization escaped the sealed study')
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if freeze['files'].get(manifest['path']) != actual or manifest['sha256'] != actual:
        raise ValueError('Initial added weights differ from their sealed source')
    state = torch.load(path, weights_only=True, map_location='cpu')
    if (state['study_sha256'] != manifest['source_study_sha256']
            or state['index_sha256'] != manifest['source_index_sha256']
            or state['completed'] != manifest['source_completed']
            or len(state['history']) != state['completed']):
        raise ValueError('Initial added weights belong to another completed source')
    for tensor in state['model'].values():
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError('Initial added weights must be finite FP32 tensors')
    model.load_state_dict(state['model'], strict=True)
    return {'source_study_sha256': state['study_sha256'],
            'source_completed': state['completed'], 'source_sha256': actual,
            'optimizer': 'fresh for the new sealed inputs'}


def fit_answers(root: Any, reader: Any, unpack: Any, config: Any, steps: Any = 24, rate: Any = .001, batch_size: Any = 4, device: Any = 'cuda',
                constrain_replay: Any = False, frontend_rate: Any = None) -> Any:
    """Train only added parameters after measured native forward/backward admission."""
    root = Path(root).resolve()
    study_sha, expected = verify_scope(root)
    index_path = root / 'answer-index.json'
    index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    index = json.loads(index_path.read_text())
    if index['role'] != 'train' or index['study_sha256'] != study_sha:
        raise ValueError('Answer index differs from frozen study')
    verify_examples(index['examples'], expected)
    pilot = json.loads((root / 'answer-probe.json').read_text())
    if pilot.get('gradient_qualified') is not True or pilot.get('study_sha256') != study_sha:
        raise ValueError('Native frozen-suffix backward qualification is required')
    if pilot.get('answer_forward_qualified') is not True:
        raise ValueError('Native final-answer forward comparison is required')
    freeze = json.loads((root / 'study-freeze.json').read_text())
    candidate = freeze['candidates'][0]
    if len(candidate['layers']) != 1 or candidate['layers'][0] not in (11, 23, 35):
        raise ValueError('Answer study requires one declared DSA placement')
    layer = candidate['layers'][0]
    width = candidate['total_added_shared_width']
    positive = []
    negative = []
    for example in index['examples']:
        if example['positive']:
            positive.append(example)
        else:
            negative.append(example)
    if not positive or not negative:
        raise ValueError('Answer fitting requires positive and preservation replay')
    if not 1 <= batch_size <= len(positive):
        raise ValueError('Answer batch size differs from admitted positive count')
    positive = interleave_answers(positive)
    model = AddedSharedNeurons(config['hidden_size'], width, streams=4).to(device)
    if frontend_rate is None:
        frontend_rate = rate
    if rate <= 0 or frontend_rate <= 0:
        raise ValueError('Answer learning rates must be positive')
    if frontend_rate == rate:
        optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    else:
        optimizer = torch.optim.Adam([
            {'params': [model.gate.weight, model.up.weight], 'lr': frontend_rate},
            {'params': [model.down.weight], 'lr': rate}])
    basis = None
    if constrain_replay:
        basis = replay_basis(root, negative, layer, device)
    recipe = {'constrain_replay': constrain_replay,
              'replay_rank': 0 if basis is None else basis.shape[1],
              'answer_order': 'proportional_group_progress_v1'}
    suffix = FrozenSuffix(reader, unpack, config, device)
    checkpoint_path = root / 'answer-optimizer.pt'
    history = []
    completed = 0
    initialization = None
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, weights_only=True, map_location=device)
        if state['study_sha256'] != study_sha or state['index_sha256'] != index_sha:
            raise ValueError('Resume checkpoint belongs to other training inputs')
        previous_recipe = state.get('recipe', {'constrain_replay': False, 'replay_rank': 0})
        if previous_recipe != recipe:
            raise ValueError('Resume checkpoint belongs to another replay constraint')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        for number, group in enumerate(optimizer.param_groups):
            group['lr'] = rate
            if len(optimizer.param_groups) == 2 and number == 0:
                group['lr'] = frontend_rate
        history = state['history']
        completed = state['completed']
        initialization = state.get('initialization')
    elif (root / 'initialization.json').exists():
        initialization = load_initial_shared(root, model)
        print(json.dumps({'phase': 'initialization', **initialization}), flush=True)
    else:
        # Seed added directions across all bare questions, with independent neurons.
        torch.manual_seed(37)
        with torch.no_grad():
            for number, example in enumerate(positive):
                data = load_rows(root, example, layer, device)
                key = data['input'][example['answer_start']].float()
                if basis is not None:
                    key = key - (key @ basis) @ basis.T
                if key.norm() <= 1e-6:
                    raise ValueError('Fact question has no direction outside ordinary replay')
                for neuron in range(number, width, len(positive)):
                    direction = key / key.square().sum().clamp_min(1e-8)
                    noise = torch.randn_like(direction) * 1e-5
                    model.gate.weight[neuron].copy_(direction + noise)
                    model.up.weight[neuron].copy_(direction - noise)
                if (number + 1) % 24 == 0:
                    print(json.dumps({'phase': 'seed', 'examples': number + 1}), flush=True)
    if basis is not None:
        with torch.no_grad():
            remove_replay_component(model.gate.weight, basis)
            remove_replay_component(model.up.weight, basis)
    cursor = presentation_count(history)
    for step in range(completed, completed + steps):
        began = time.monotonic()
        print(json.dumps({'phase': 'answer_backward', 'step': step + 1}), flush=True)
        examples = []
        rows = []
        preservation_rows = []
        for offset in range(batch_size):
            number = cursor + offset
            example = positive[number % len(positive)]
            replay = negative[number % len(negative)]
            examples.append(example)
            rows.append(load_rows(root, example, layer, device))
            preservation_rows.append(load_rows(root, replay, layer, device))
        optimizer.zero_grad(set_to_none=True)
        answer_loss, additions = final_answer_batch(model, rows, suffix, examples, layer)
        preservation_losses = []
        prefix_losses = []
        for example, data, preservation, added in zip(examples, rows, preservation_rows,
                                                      additions, strict=True):
            scale = data['base_output'].float().square().mean().clamp_min(1e-4)
            negative_output = model(preservation['input'].float())
            preservation_losses.append(negative_output.square().mean() / scale)
            prefix_losses.append(added[:example['answer_start']].square().mean() / scale)
        preserve_loss = torch.stack(preservation_losses).mean()
        prefix_loss = torch.stack(prefix_losses).mean()
        loss = answer_loss + 10 * preserve_loss + 5 * prefix_loss
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite final-answer objective')
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise ValueError('Missing or nonfinite final-answer gradient')
        if basis is not None:
            with torch.no_grad():
                remove_replay_component(model.gate.weight.grad, basis)
                remove_replay_component(model.up.weight.grad, basis)
        down_gradient = model.down.weight.grad
        if down_gradient is None:
            raise ValueError('Missing down projection gradient')
        stream_norms = down_gradient.reshape(4, config['hidden_size'], width).norm(dim=(1, 2))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if basis is not None:
            with torch.no_grad():
                remove_replay_component(model.gate.weight, basis)
                remove_replay_component(model.up.weight, basis)
        cursor += batch_size
        row = {'step': step + 1, 'example_ids': [example['id'] for example in examples],
               'answer_ce': float(answer_loss.detach()),
               'answer_objective': 'all_answer_tokens_mean',
               'preservation': float(preserve_loss.detach()), 'prefix': float(prefix_loss.detach()),
               'stream_gradient_norms': stream_norms.detach().cpu().tolist(),
               'seconds': time.monotonic() - began, 'learning_rate': rate,
               'frontend_learning_rate': frontend_rate, 'recipe': recipe,
               'cuda_peak_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else 0}
        history.append(row)
        print(json.dumps(row), flush=True)
        if (step + 1) % 4 == 0 or step + 1 == completed + steps:
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'study_sha256': study_sha, 'index_sha256': index_sha,
                     'recipe': recipe,
                     'initialization': initialization,
                     'completed': step + 1, 'history': history}
            temporary = checkpoint_path.with_suffix('.tmp')
            torch.save(state, temporary)
            temporary.replace(checkpoint_path)
            site = {'gate': model.gate.weight.detach().cpu(), 'up': model.up.weight.detach().cpu(),
                    'down': model.down.weight.detach().cpu()}
            selected = []
            for rank in range(2):
                artifact = {'schema_version': 1, 'mechanism': 'shared_ffn_four_streams_v2',
                            'rank': rank, 'world': 2, 'base_frozen': True,
                            'study_sha256': study_sha, 'index_sha256': index_sha,
                            'initialization': initialization,
                            'layers': {layer: partition_shared(site, rank)}}
                path = root / 'pair' / ('answer-step' + str(step + 1) + '-rank' + str(rank) + '.pt')
                torch.save(artifact, path)
                selected.append({'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            manifest = [{**candidate, 'id': candidate['id'] + '-answer-step' + str(step + 1),
                         'selected': selected, 'answer_steps': step + 1,
                         'initialization': initialization}]
            body = json.dumps(manifest, indent=2) + '\n'
            (root / ('fit-manifest-answer-step' + str(step + 1) + '.json')).write_text(body)
            (root / 'fit-manifest-answer.json').write_text(body)
            (root / 'answer-history.json').write_text(json.dumps(history, indent=2) + '\n')
    return history
