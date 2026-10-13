"""Tripwires for final-answer gradient reach and paired capture admission."""

import hashlib
import json

import pytest
import torch

from glm_rsi_dream.neurons.answer import (
    final_answer_loss,
    interleave_answers,
    load_initial_shared,
    load_rows,
    presentation_count,
    remove_replay_component,
)
from glm_rsi_dream.neurons.fit import AddedSharedNeurons


def test_final_answer_loss_reaches_every_stream_and_preserves_frozen_head():
    torch.manual_seed(31)
    class Suffix:
        def __init__(self):
            self.weight = torch.randn(5, 4)
        def __call__(self, streams, start):
            assert start == 36
            hidden = streams[:, 0] + 2 * streams[:, 1]
            hidden = hidden + 3 * streams[:, 2] + 4 * streams[:, 3]
            return torch.nn.functional.linear(hidden, self.weight)
    suffix = Suffix()
    frozen = suffix.weight.clone()
    data = {'input': torch.randn(3, 4), 'before': torch.randn(3, 4, 4),
            'base_output': torch.randn(3, 4), 'post': torch.ones(3, 4),
            'combine': torch.eye(4).expand(3, -1, -1)}
    model = AddedSharedNeurons(4, 6, streams=4)
    loss, _ = final_answer_loss(model, data, suffix, {'answer_start': 1, 'target_ids': [2, 1]}, 35)
    loss.backward()
    norms = model.down.weight.grad.reshape(4, 4, 6).norm(dim=(1, 2))
    assert torch.all(norms > 0)
    assert suffix.weight.grad is None
    assert torch.equal(suffix.weight, frozen)


def captured_example(root, mismatch=False):
    chunk = {'positions': [0, 1], 'input': torch.ones(2, 4),
             'before': torch.ones(2, 4, 4), 'residual': torch.ones(2, 4, 4),
             'base_output': torch.ones(2, 4), 'post': torch.ones(2, 4),
             'combine': torch.eye(4).expand(2, -1, -1)}
    entries = []
    for rank in range(2):
        local = dict(chunk)
        if mismatch and rank == 1:
            local['input'] = torch.zeros(2, 4)
        packet = {'schema_version': 1, 'rank': rank, 'world': 2,
                  'attention_memory': False, 'layers': {35: [local]}}
        path = root / ('rank' + str(rank) + '.pt')
        torch.save(packet, path)
        entries.append({'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return {'captures': entries, 'token_ids': [3, 4]}


def test_answer_capture_requires_identical_full_states_from_both_ranks(tmp_path):
    example = captured_example(tmp_path)
    rows = load_rows(tmp_path, example, 35, device='cpu')
    assert rows['before'].shape == (2, 4, 4)
    example['captures'].pop()
    with pytest.raises(ValueError, match='both native ranks'):
        load_rows(tmp_path, example, 35, device='cpu')


def test_answer_capture_rejects_rank_disagreement_and_missing_prefix(tmp_path):
    example = captured_example(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match='states disagree'):
        load_rows(tmp_path, example, 35, device='cpu')
    example = captured_example(tmp_path)
    example['token_ids'].append(5)
    with pytest.raises(ValueError, match='fresh full token sequence'):
        load_rows(tmp_path, example, 35, device='cpu')


def test_replay_constraint_preserves_plain_frontend_after_answer_updates():
    torch.manual_seed(51)
    model = AddedSharedNeurons(8, 6, streams=4)
    basis = torch.eye(8)[:, :3]
    ordinary = torch.zeros(5, 8)
    ordinary[:, :3] = torch.randn(5, 3)
    facts = torch.randn(5, 8)
    target = torch.randn(5, 4, 8)
    with torch.no_grad():
        remove_replay_component(model.gate.weight, basis)
        remove_replay_component(model.up.weight, basis)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(facts) - target).square().mean()
        loss.backward()
        with torch.no_grad():
            remove_replay_component(model.gate.weight.grad, basis)
            remove_replay_component(model.up.weight.grad, basis)
        optimizer.step()
        with torch.no_grad():
            remove_replay_component(model.gate.weight, basis)
            remove_replay_component(model.up.weight, basis)
        assert torch.count_nonzero(model(ordinary)) == 0
    assert model(facts).norm() > 0
    for stream in model(facts).unbind(1):
        assert stream.norm() > 0


def test_presentation_cursor_survives_resumed_batch_size_change():
    history = [{'example_ids': ['a', 'b', 'c', 'd']},
               {'example_ids': ['e', 'f']}]
    assert presentation_count(history) == 6


def test_unequal_fact_groups_remain_present_through_last_batch():
    examples = []
    for name, count in [('alpha', 16), ('beta', 8), ('gamma', 8)]:
        for number in range(count):
            examples.append({'id': name + '-train-' + str(number)})
    ordered = interleave_answers(examples)
    assert sorted(row['id'] for row in ordered) == sorted(row['id'] for row in examples)
    for start in range(0, len(ordered), 8):
        counts = dict.fromkeys(['alpha', 'beta', 'gamma'], 0)
        for row in ordered[start:start + 8]:
            name = row['id'].split('-train-')[0]
            counts[name] += 1
        assert counts == {'alpha': 4, 'beta': 2, 'gamma': 2}
    assert interleave_answers([]) == []


def test_factual_and_ending_positions_receive_equal_answer_weight():
    class Suffix:
        def __init__(self):
            self.logits = torch.zeros(3, 5, requires_grad=True)
        def __call__(self, streams, start):
            return self.logits + streams.sum() * 0
    suffix = Suffix()
    data = {'input': torch.zeros(3, 4), 'before': torch.zeros(3, 4, 4),
            'base_output': torch.zeros(3, 4), 'post': torch.ones(3, 4),
            'combine': torch.eye(4).expand(3, -1, -1)}
    model = AddedSharedNeurons(4, 6, streams=4)
    loss, _ = final_answer_loss(model, data, suffix,
                                {'answer_start': 0, 'target_ids': [1, 2, 3]}, 35)
    loss.backward()
    norms = suffix.logits.grad.norm(dim=1)
    assert torch.allclose(norms, norms[0].expand_as(norms))


def initial_shared_study(root):
    source = AddedSharedNeurons(4, 6, streams=4)
    with torch.no_grad():
        source.down.weight.normal_()
    state = {'model': source.state_dict(), 'study_sha256': 'a' * 64,
             'index_sha256': 'b' * 64, 'completed': 12, 'history': [{}] * 12,
             'optimizer': {'old_inputs': True}}
    path = root / 'initial.pt'
    torch.save(state, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'role': 'added shared weights initialization', 'path': path.name,
                'sha256': digest, 'source_study_sha256': state['study_sha256'],
                'source_index_sha256': state['index_sha256'], 'source_completed': 12}
    manifest_path = root / 'initialization.json'
    manifest_path.write_text(json.dumps(manifest))
    freeze = {'files': {'initialization.json': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                        path.name: digest}}
    (root / 'study-freeze.json').write_text(json.dumps(freeze))
    return source, manifest


def test_initial_shared_weights_preserve_all_streams_for_new_input_study(tmp_path):
    source, _ = initial_shared_study(tmp_path)
    model = AddedSharedNeurons(4, 6, streams=4)
    receipt = load_initial_shared(tmp_path, model)
    inputs = torch.randn(3, 4)
    assert torch.equal(source(inputs), model(inputs))
    for stream in model(inputs).unbind(1):
        assert stream.norm() > 0
    assert receipt['source_completed'] == 12
    assert receipt['optimizer'] == 'fresh for the new sealed inputs'


def test_initial_shared_weights_reject_other_source_and_unsealed_data(tmp_path):
    _, manifest = initial_shared_study(tmp_path)
    model = AddedSharedNeurons(4, 6, streams=4)
    manifest['source_completed'] = 8
    path = tmp_path / 'initialization.json'
    path.write_text(json.dumps(manifest))
    freeze_path = tmp_path / 'study-freeze.json'
    freeze = json.loads(freeze_path.read_text())
    freeze['files']['initialization.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
    freeze_path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match='another completed source'):
        load_initial_shared(tmp_path, model)
    manifest['source_completed'] = 12
    manifest['sha256'] = '0' * 64
    path.write_text(json.dumps(manifest))
    freeze['files']['initialization.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
    freeze_path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match='sealed source'):
        load_initial_shared(tmp_path, model)
