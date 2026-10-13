"""A pocket-off epoch must exclude resident snapshots and automatic admission."""

import json
from types import SimpleNamespace

import pytest

from glm_rsi_dream.neurons import runtime


def test_off_gate_binds_an_owner_and_rejects_incomplete_value(tmp_path):
    w = SimpleNamespace(meta={runtime.KEY: {'root': tmp_path}})
    assert not runtime.attention_disabled(w)
    path = tmp_path / 'attention-disabled.json'
    path.write_text(json.dumps({'all_attention_disabled': True}))
    with pytest.raises(ValueError, match='gate'):
        runtime.attention_disabled(w)
    path.write_text(json.dumps({'all_attention_disabled': True, 'owner': 'placement-study'}))
    assert runtime.attention_disabled(w)


def test_runtime_rejects_a_retained_attention_snapshot(tmp_path):
    (tmp_path / 'attention-disabled.json').write_text(json.dumps({
        'all_attention_disabled': True, 'owner': 'placement-study'}))
    state = {'root': tmp_path, 'active': {}, 'snapshot': object()}
    w = SimpleNamespace(meta={runtime.KEY: state})
    with pytest.raises(ValueError, match='memory state'):
        runtime.status(w)


def test_global_neurons_disable_cached_decode_selection(tmp_path):
    from tensorfold_dream.eval_native import has_main

    state = {'root': tmp_path}
    w = SimpleNamespace(meta={runtime.KEY: state})
    b = SimpleNamespace(_eval_pieces=[])
    assert not has_main(w, b)
    (tmp_path / 'neurons-current.json').write_text('{}')
    assert has_main(w, b)


def test_runtime_rejects_an_old_engine_reset_epoch(tmp_path):
    (tmp_path / 'attention-disabled.json').write_text(json.dumps({
        'all_attention_disabled': True, 'owner': 'placement-study', 'epoch': 'current'}))
    (tmp_path / 'automatic.json').write_text(json.dumps({'status': 'disabled'}))
    state = {'root': tmp_path, 'active': {}, 'pair_sha256': 'a' * 64}
    w = SimpleNamespace(meta={runtime.KEY: state}, rank=0, world=2)
    runtime.record_clean_start(w, 'old')
    with pytest.raises(ValueError, match='epoch receipt'):
        runtime.status(w)
