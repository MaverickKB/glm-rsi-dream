import base64
import copy
from unittest.mock import patch

import pytest

from glm_rsi_dream.agent import clone, stage


def test_clone_preserves_original_and_disables_candidate_restart():
    original = {'Id': 'original-example', 'Image': 'sha256:example',
                'Config': {'Image': 'original-tag', 'Env': ['TF_GLM_HC_SPLIT=1', 'TF_GLM_PREFILL_OVERLAP=1'],
                           'Labels': {}, 'Cmd': ['serve']},
                'HostConfig': {'Binds': ['/example/models:/models:ro'], 'RestartPolicy': {'Name': 'always'}}}
    before = copy.deepcopy(original)
    identity = {name: original[name] for name in ['Config', 'HostConfig', 'Image']}
    created = {}
    def docker(method, path, body):
        created.update(copy.deepcopy(body))
        return {'Id': 'candidate-example'}
    def inspect(container):
        if container == 'original-example':
            return copy.deepcopy(original)
        config = copy.deepcopy(created)
        host = config.pop('HostConfig')
        return {'Image': original['Image'], 'Config': config, 'HostConfig': host}
    spec = {'original': 'original-example', 'identity': identity, 'run_id': 'cpu-example', 'rank': 1,
            'root': '/example/run', 'site_dir': '/example/site', 'engine_sha256': 'a' * 64}
    with patch('glm_rsi_dream.agent.inspect', side_effect=inspect), patch('glm_rsi_dream.agent.docker', side_effect=docker):
        result = clone(spec)
    assert result['validated']
    assert original == before
    assert created['HostConfig']['RestartPolicy']['Name'] == 'no'
    assert 'TF_GLM_HC_SPLIT=0' in created['Env']
    assert 'TF_GLM_PREFILL_OVERLAP=0' in created['Env']
    assert not any('--port' in arg for arg in created['Cmd'])


def test_stage_rejects_a_member_with_no_bytes(tmp_path, monkeypatch):
    import io
    import tarfile
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("note.txt")
        info.size = 4
        archive.addfile(info, io.BytesIO(b"note"))
    monkeypatch.setattr(tarfile.TarFile, "extractfile", lambda self, member: None)
    with pytest.raises(ValueError, match="regular staged files"):
        stage(tmp_path / "run", base64.b64encode(buffer.getvalue()).decode())


def test_stage_rejects_path_escape_and_links(tmp_path):
    import io
    import tarfile
    for name, link in [('../escape', False), ('link', True)]:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
            info = tarfile.TarInfo(name)
            if link:
                info.type = tarfile.SYMTYPE
                info.linkname = '../outside'
                archive.addfile(info)
            else:
                info.size = 1
                archive.addfile(info, io.BytesIO(b'x'))
        with pytest.raises(ValueError):
            stage(tmp_path / 'run', base64.b64encode(buffer.getvalue()).decode())
    assert not (tmp_path / 'escape').exists()


def test_worker_generation_listener_check_reads_ipv4_and_ipv6():
    from pathlib import Path

    from glm_rsi_dream.agent import dispatch
    empty = '  sl  local_address rem_address st\n'
    listener = '  0: 00000000:22B8 00000000:0000 0A\n'
    for family in ['tcp', 'tcp6']:
        def read(path):
            if path.name == family:
                return empty + listener
            return empty
        with patch.object(Path, 'read_text', read):
            with pytest.raises(RuntimeError, match='Worker generation listener'):
                dispatch({'action': 'no_listener', 'port': 8888})
    with patch.object(Path, 'read_text', return_value=empty):
        assert dispatch({'action': 'no_listener', 'port': 8888})['absent']
