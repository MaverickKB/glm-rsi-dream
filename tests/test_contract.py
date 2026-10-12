import json
from pathlib import Path
from unittest.mock import patch

import pytest

from glm_rsi_dream.cli import main
from glm_rsi_dream.config import load_config
from glm_rsi_dream.dataset import admitted, close_fitting, freeze, validate, verify
from glm_rsi_dream.files import read_json, write_json
from glm_rsi_dream.grader import grade
from glm_rsi_dream.grading import assess, decision, parse_verdict
from glm_rsi_dream.safety import MemoryGate, SafetyError, require_parity

ROOT = Path(__file__).resolve().parents[1]


def fixtures():
    lesson = read_json(ROOT / 'examples/synthetic/lesson.json')
    evaluation = read_json(ROOT / 'examples/synthetic/eval.json')
    return lesson, evaluation


def reply(text, finish='stop'):
    return {'choices': [{'message': {'content': text}, 'finish_reason': finish}],
            'usage': {'completion_tokens': 2}}


def test_demo_split_routing_and_disjointness():
    lesson, evaluation = fixtures()
    validate(lesson, evaluation)
    assert len(lesson['training']) == 72
    assert len(lesson['development']) == 12
    assert len(evaluation['heldout']) == 36
    assert admitted('What is the Linden workshop tablet model?', lesson['routes'])
    assert not admitted('What is the Maple workshop tablet model?', lesson['routes'])
    evaluation['heldout'][0]['prompt'] = lesson['training'][0]['prompt']
    with pytest.raises(ValueError, match='overlap'):
        validate(lesson, evaluation)


def test_dry_run_is_offline_and_has_no_side_effects(tmp_path):
    for command in ['run', 'ask', 'restore', 'status']:
        with patch('subprocess.run', side_effect=AssertionError('process opened')), patch('urllib.request.urlopen', side_effect=AssertionError('network opened')):
            assert main(['--config', str(ROOT / 'config.example.json'), '--run', str(tmp_path / 'absent'),
                         '--lesson', str(ROOT / 'examples/synthetic/lesson.json'),
                         '--eval', str(ROOT / 'examples/synthetic/eval.json'), '--dry-run', command]) == 0
    assert not (tmp_path / 'absent').exists()


def test_live_placeholders_and_mac_are_rejected_before_network(tmp_path):
    with pytest.raises(ValueError, match='placeholders'):
        load_config(ROOT / 'config.example.json', live=True)
    config = load_config(ROOT / 'config.example.json')
    config['hosts'][0]['ssh'] = 'example-head'
    config['hosts'][1]['ssh'] = 'example-worker'
    config['hosts'][0]['original_container'] = 'baseline-head'
    config['hosts'][1]['original_container'] = 'baseline-worker'
    config['api_url'] = 'http://example-head:8888'
    config['safety']['protected_urls'] = ['http://example-service:8080/health']
    config['baseline'] = {}
    path = tmp_path / 'config.json'
    write_json(path, config)
    with patch('platform.system', return_value='Darwin'), patch('subprocess.run', side_effect=AssertionError('opened')):
        assert main(['--config', str(path), '--live', 'run']) == 1


def test_freeze_detects_changes_and_holdout_marker_closes_fitting(tmp_path):
    from types import SimpleNamespace
    cluster = SimpleNamespace(run_dir=tmp_path / 'new', config=load_config(ROOT / 'config.example.json'))
    freeze(cluster, ROOT / 'examples/synthetic/lesson.json', ROOT / 'examples/synthetic/eval.json')
    verify(cluster)
    close_fitting(cluster, [{'sha256': 'a' * 64}])
    with pytest.raises(SafetyError, match='already opened'):
        close_fitting(cluster, [])
    changed = read_json(cluster.run_dir / 'lesson.json')
    changed['facts'][0]['statement'] += ' extra'
    write_json(cluster.run_dir / 'lesson.json', changed)
    with pytest.raises(SafetyError, match='dataset changed'):
        verify(cluster)


def test_glm_grader_rejects_unsupported_extras_despite_alias():
    from types import SimpleNamespace
    cluster = SimpleNamespace(config={'capture_max_tokens': 512}, body=lambda prompt: {'prompt': prompt})
    cluster.infer = lambda body: (reply('{"correct":true,"grounded":false,"reason":"unsupported extra"}'), [])
    task = {'prompt': 'Name the helper', 'aliases': [['Orbit']]}
    result = assess(cluster, ['The helper is Orbit.'], task, 'Orbit also cooks dinner.')
    assert result['assertion']
    assert not result['grounded']
    assert not grade('The helper is not Orbit.', task)
    assert not grade("I don't know, maybe Orbit.", task)


@pytest.mark.parametrize('value', ['{"correct":"true","grounded":true,"reason":"ok"}',
                                    '{"correct":true,"grounded":true}', 'not JSON'])
def test_malformed_grader_fails_closed(value):
    with pytest.raises((SafetyError, json.JSONDecodeError)):
        parse_verdict(value)


def test_decision_requires_gain_per_fact_and_control_preservation():
    acceptance = load_config(ROOT / 'config.example.json')['acceptance']
    rows = []
    for i in range(10):
        rows.append({'fact_id': str(i % 2), 'memory': {'grounded': True}, 'baseline': {'grounded': False}})
    controls = [{'byte_identical': True, 'zero_installation': True}]
    assert decision(rows, controls, acceptance)['verdict'] == 'keep'
    for row in rows:
        row['baseline']['grounded'] = True
    assert decision(rows, controls, acceptance)['verdict'] == 'reject'
    for row in rows:
        row['baseline']['grounded'] = False
    controls[0]['zero_installation'] = False
    assert decision(rows, controls, acceptance)['verdict'] == 'reject'


def test_memory_swap_and_exact_byte_gates():
    policy = load_config(ROOT / 'config.example.json')['safety']
    gate = MemoryGate(policy)
    with pytest.raises(SafetyError, match='memory floor'):
        gate.observe(0, {'available': 3_999_999_999, 'time': 1, 'swap_out_bytes': 0})
    gate = MemoryGate(policy)
    for seconds in range(30):
        gate.observe(1, {'available': 5_000_000_000, 'time': seconds, 'swap_out_bytes': seconds * 20_000_000})
    with pytest.raises(SafetyError, match='swap-out'):
        gate.observe(1, {'available': 5_000_000_000, 'time': 30, 'swap_out_bytes': 600_000_000})
    require_parity([reply('Exact\n')], [reply('Exact\n')])
    for text in ['exact\n', 'Exact', 'Exact ']:
        with pytest.raises(SafetyError):
            require_parity([reply('Exact\n')], [reply(text)])
    with pytest.raises(SafetyError):
        require_parity([reply('partial', 'length')], [reply('partial', 'length')])


def test_atomic_private_files_survive_concurrent_writes(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    path = tmp_path / 'private' / 'state.json'
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for i in range(80):
            futures.append(executor.submit(write_json, path, {'value': i}))
        for future in futures:
            future.result()
    assert isinstance(read_json(path)['value'], int)
    assert path.stat().st_mode & 0o777 == 0o600
    assert not list(path.parent.glob('*.tmp'))


def test_status_excludes_preserved_environment(tmp_path, capsys):
    write_json(tmp_path / 'state.json', {'phase': 'restored', 'originals': [{'Config': {'Env': ['TOKEN=private-example']}}]})
    assert main(['--run', str(tmp_path), 'status']) == 0
    assert 'private-example' not in capsys.readouterr().out
