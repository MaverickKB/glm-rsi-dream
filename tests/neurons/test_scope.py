"""A capture's row labels cannot override the frozen training membership."""
import hashlib
import json

import pytest

from glm_rsi_dream.files import digest
from glm_rsi_dream.neurons.scope import verify_examples, verify_scope


@pytest.mark.parametrize('examples', [
    [{'id': 'reserved-fresh', 'positive': True}],
    [{'id': 'training', 'positive': False}],
    [{'id': 'training', 'positive': 'true'}],
    [{'id': 'training', 'positive': True}, {'id': 'training', 'positive': True}],
    [],
])
def test_capture_scope_rejects_reserved_relabelled_duplicate_or_missing_rows(examples):
    with pytest.raises(ValueError):
        verify_examples(examples, {'training': True})


def test_scope_seals_additional_native_framing_inputs(tmp_path):
    contents = {'train.json': [{'id': 'fictional-train', 'positive': True}],
                'dev.json': [], 'fresh.json': [], 'comparison.json': [],
                'pockets.json': [], 'ordinary-source.json': [], 'protected.json': [],
                'user-only-prefixes.json': {'examples': [{'id': 'fictional-train', 'prompt_ids': [1, 2]}]}}
    files = {}
    for name, value in contents.items():
        path = tmp_path / name
        path.write_text(json.dumps(value))
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    freeze = {'before_capture_and_fitting': True, 'files': files}
    freeze['sha256'] = digest(freeze)
    (tmp_path / 'study-freeze.json').write_text(json.dumps(freeze))
    verify_scope(tmp_path)
    (tmp_path / 'user-only-prefixes.json').write_text('{"examples": []}')
    with pytest.raises(ValueError, match='user-only-prefixes'):
        verify_scope(tmp_path)


def test_inherited_capture_keeps_original_inputs_beside_new_initialization(tmp_path):
    source = tmp_path / 'capture-origin'
    source.mkdir()
    (source / 'initialization.json').write_text('{"source": "earlier"}')
    original = {'files': {'initialization.json': hashlib.sha256(
        (source / 'initialization.json').read_bytes()).hexdigest()}}
    original['sha256'] = digest(original)
    (tmp_path / 'study-freeze-original.json').write_text(json.dumps(original))
    index = tmp_path / 'capture-index-original.json'
    index.write_text('{"examples": []}')
    contents = {'train.json': [{'id': 'fictional-train', 'positive': True}],
                'dev.json': [], 'fresh.json': [], 'comparison.json': [],
                'pockets.json': [], 'ordinary-source.json': [], 'protected.json': [],
                'initialization.json': {'source': 'later'}}
    files = {}
    for name, value in contents.items():
        path = tmp_path / name
        path.write_text(json.dumps(value))
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    freeze = {'before_fitting': True, 'files': files, 'inherited_capture': {
        'source_files_path': 'capture-origin', 'membership_audited': True,
        'original_index_sha256': hashlib.sha256(index.read_bytes()).hexdigest(),
        'original_study_sha256': original['sha256']}}
    freeze['sha256'] = digest(freeze)
    target = tmp_path / 'study-freeze.json'
    target.write_text(json.dumps(freeze))
    verify_scope(tmp_path)
    freeze.pop('sha256')
    freeze['inherited_capture']['source_files_path'] = '..'
    freeze['sha256'] = digest(freeze)
    target.write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match='escaped its root'):
        verify_scope(tmp_path)
    freeze.pop('sha256')
    freeze['inherited_capture']['source_files_path'] = 'capture-origin'
    freeze['sha256'] = digest(freeze)
    target.write_text(json.dumps(freeze))
    (source / 'initialization.json').write_text('{"source": "changed"}')
    with pytest.raises(ValueError, match='Inherited capture input changed'):
        verify_scope(tmp_path)
