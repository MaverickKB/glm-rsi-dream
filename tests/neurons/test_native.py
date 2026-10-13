"""CPU checks of native artifact ownership and pocket-independent application."""

import hashlib
from types import SimpleNamespace

import pytest
import torch

from glm_rsi_dream.neurons import native


def test_native_ffn_uses_only_hidden_state_with_attention_disabled(tmp_path, monkeypatch):
    site = {"gate": torch.eye(4), "up": torch.eye(4), "down": torch.eye(4)}
    value = {"mechanism": "shared_ffn_swiglu_v1", "rank": 0, "world": 2,
             "base_frozen": True, "layers": {19: site}}
    path = tmp_path / "new.pt"
    torch.save(value, path)
    selected = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    record = {"policy": {"neurons": selected, "use_memory": False}}
    monkeypatch.setattr(native, "records", lambda w, b: [(record, None, 0, 2)])
    w = SimpleNamespace(meta={native.KEY: {"root": tmp_path}}, rank=0, device="cpu",
                        cfg=SimpleNamespace(hidden=4, limit=10))
    b = SimpleNamespace(normed=torch.tensor([[1., -2., 3., -4.], [-1., -2., -3., -4.]]),
                        sy=torch.zeros(2, 4))
    native.shared(SimpleNamespace(index=19), w, b, 0, 2)
    expected = torch.nn.functional.silu(b.normed) * b.normed
    torch.testing.assert_close(b.sy, expected)
    assert record["neuron_sha256"] == selected["sha256"]
    assert not record["policy"]["use_memory"]
    record["policy"]["neurons_off"] = True
    b.sy.zero_()
    native.shared(SimpleNamespace(index=19), w, b, 0, 2)
    assert torch.count_nonzero(b.sy) == 0


def test_native_rejects_another_ranks_artifact(tmp_path):
    path = tmp_path / "wrong.pt"
    torch.save({"mechanism": "shared_ffn_swiglu_v1", "rank": 1, "world": 2,
                "base_frozen": True, "layers": {}}, path)
    selected = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    w = SimpleNamespace(meta={native.KEY: {"root": tmp_path}}, rank=0, device="cpu")
    with pytest.raises(ValueError, match="ownership"):
        native.artifact(w, selected)


def test_native_rejects_mutated_capture_artifact(tmp_path):
    path = tmp_path / "changed.pt"
    path.write_bytes(b"changed")
    selected = {"path": path.name, "sha256": "a" * 64}
    w = SimpleNamespace(meta={native.KEY: {"root": tmp_path}}, rank=0, device="cpu")
    with pytest.raises(ValueError, match="hash"):
        native.artifact(w, selected)


def test_native_four_stream_write_reduces_shards_and_uses_each_post_weight(tmp_path, monkeypatch):
    from glm_rsi_dream.neurons.reference import hyper_post

    torch.manual_seed(27)
    site = {'gate': torch.randn(2, 4), 'up': torch.randn(2, 4), 'down': torch.randn(16, 2)}
    value = {'mechanism': 'shared_ffn_four_streams_v2', 'rank': 0, 'world': 2,
             'base_frozen': True, 'layers': {35: site}}
    path = tmp_path / 'four.pt'
    torch.save(value, path)
    selected = {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    record = {'policy': {'neurons': selected, 'use_memory': False}}
    monkeypatch.setattr(native, 'records', lambda w, b: [(record, None, 0, 2)])
    def cpu_mix(before, output, branch, post, combine):
        mixed = hyper_post(before.reshape(2, 4, 4), branch, post, combine)
        output.copy_(mixed.reshape_as(output))
    monkeypatch.setattr(native, 'mix_branch', cpu_mix)
    peer = torch.randn(2, 4, 4)
    class Communicator:
        def all_gather(self, partial, output):
            partial = partial.reshape(2, 4, 4)
            output = output.reshape(2, 2, 4, 4)
            output[0].copy_(partial)
            output[1].copy_(peer)
    w = SimpleNamespace(meta={native.KEY: {'root': tmp_path}}, rank=0, world=2,
                        device='cpu', comm=Communicator(), cfg=SimpleNamespace(hidden=4, limit=10))
    b = SimpleNamespace(normed=torch.randn(2, 4), x=torch.randn(2, 16).bfloat16(),
                        post=torch.rand(2, 4), comb=torch.rand(2, 4, 4), prefill=False,
                        sy=torch.zeros(2, 4))
    before = b.x.reshape(2, 4, 4).clone()
    layer = SimpleNamespace(index=35)
    native.before_ffn(layer, w, b, 2)
    native.shared(layer, w, b, 0, 2)
    gathered = torch.randn(2, 2, 4)
    branch = gathered[0] + gathered[1]
    expected = hyper_post(before, branch[:, None] + b._dream_delta + peer, b.post, b.comb)
    native.after_ffn(layer, w, b, 2, gathered)
    torch.testing.assert_close(b.x.reshape(2, 4, 4), expected, rtol=0, atol=0)
    assert torch.count_nonzero(b.sy) == 0
    assert b._dream_delta is None
    record['policy']['neurons_off'] = True
    native.before_ffn(layer, w, b, 2)
    assert b._dream_delta is None


def test_decode_capture_retains_prefill_and_generated_positions(monkeypatch):
    spec = {'layers': [35], 'rows': list(range(4)), 'answer_trace': True,
            'decode_trace': True}
    record = {'policy': {'neuron_capture': spec}}
    state = SimpleNamespace(pos=0)
    w = SimpleNamespace(world=2, cfg=SimpleNamespace(hidden=4))
    b = SimpleNamespace(prefill=True, split=None, normed=torch.randn(2, 4),
                        post=torch.rand(2, 4), x=torch.randn(2, 16),
                        comb=torch.rand(2, 4, 4))
    monkeypatch.setattr(native, 'selected_records', lambda w, b: [])
    pieces = [(record, state, 0, 2)]
    monkeypatch.setattr(native, 'records', lambda w, b: pieces)
    layer = SimpleNamespace(index=35)
    gathered = torch.randn(2, 2, 4)
    native.before_ffn(layer, w, b, 2)
    native.after_ffn(layer, w, b, 2, gathered)
    state.pos = 2
    b.prefill = False
    pieces[0] = (record, state, 1, 1)
    native.before_ffn(layer, w, b, 2)
    native.after_ffn(layer, w, b, 2, gathered)
    chunks = record['neuron_chunks'][35]
    assert chunks[0]['positions'] == [0, 1]
    assert chunks[1]['positions'] == [2]
    torch.testing.assert_close(chunks[1]['input'], b.normed[1:2])
    torch.testing.assert_close(chunks[1]['base_output'], gathered[:, 1:2].sum(dim=0))
    assert chunks[1]['before'].shape == (1, 4, 4)
    assert chunks[1]['residual'].shape == (1, 4, 4)
    del spec['decode_trace']
    state.pos = 3
    native.before_ffn(layer, w, b, 2)
    native.after_ffn(layer, w, b, 2, gathered)
    assert len(record['neuron_chunks'][35]) == 2


def test_decode_capture_bounds_total_request_rows(monkeypatch):
    spec = {'layers': [35], 'rows': [512], 'decode_trace': True}
    record = {'policy': {'neuron_capture': spec},
              'neuron_chunks': {35: [{'positions': list(range(512))}]}}
    state = SimpleNamespace(pos=512)
    w = SimpleNamespace(cfg=SimpleNamespace(hidden=4))
    b = SimpleNamespace(prefill=False, split=None)
    monkeypatch.setattr(native, 'selected_records', lambda w, b: [])
    monkeypatch.setattr(native, 'records', lambda w, b: [(record, state, 0, 1)])
    with pytest.raises(ValueError, match='bounded geometry'):
        native.before_ffn(SimpleNamespace(index=35), w, b, 1)
