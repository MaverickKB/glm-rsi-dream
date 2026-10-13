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


def test_canonical_data_directory_is_the_cache(monkeypatch, tmp_path):
    from glm_rsi_dream.agent import expand_root
    from glm_rsi_dream.cli import build_parser
    from glm_rsi_dream.config import DEFAULT_DATA_DIR, DEFAULT_RUN_DIR, shell_command
    example = read_json(ROOT / 'config.example.json')
    assert example['hosts'][0]['work_dir'] == DEFAULT_DATA_DIR
    assert example['hosts'][1]['work_dir'] == DEFAULT_DATA_DIR
    loaded = load_config(ROOT / 'config.example.json')
    assert loaded['hosts'][0]['work_dir'] == DEFAULT_DATA_DIR
    assert loaded['hosts'][1]['work_dir'] == DEFAULT_DATA_DIR
    assert build_parser().parse_args(['status']).run == DEFAULT_RUN_DIR
    monkeypatch.setenv('HOME', str(tmp_path))
    expected = str(tmp_path / '.cache/glm-rsi-dream/demo-1')
    assert expand_root(DEFAULT_RUN_DIR) == expected
    command = shell_command(['mkdir', '-p', DEFAULT_DATA_DIR + '/baseline'])
    assert command == 'mkdir -p $HOME/.cache/glm-rsi-dream/baseline'
    mounted = DEFAULT_DATA_DIR + '/baseline:/models'
    assert shell_command(['docker', 'run', '-v', mounted]) == 'docker run -v $HOME/.cache/glm-rsi-dream/baseline:/models'


def test_fact_statements_satisfy_their_alias_groups_and_not_another_fact():
    lesson, evaluation = fixtures()
    facts = {}
    for fact in lesson['facts']:
        facts[fact['id']] = fact['statement']
    splits = [('train', lesson['training']), ('dev', lesson['development']),
              ('heldout', evaluation['heldout'])]
    for split, rows in splits:
        for row in rows:
            statement = facts[row['fact_id']]
            assert grade(statement, row), split + ' ' + row['id']
            for other_id, other in facts.items():
                if other_id == row['fact_id']:
                    continue
                assert not grade(other, row), row['id'] + ' also matches ' + other_id


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
    controls = [{'byte_identical': True, 'zero_installation': True,
                 'zero_cache_writes': True, 'zero_offpath_delta': True}]
    assert decision(rows, controls, acceptance)['verdict'] == 'keep'
    for row in rows:
        row['baseline']['grounded'] = True
    assert decision(rows, controls, acceptance)['verdict'] == 'reject'
    for row in rows:
        row['baseline']['grounded'] = False
    controls[0]['zero_installation'] = False
    assert decision(rows, controls, acceptance)['verdict'] == 'reject'
    controls[0]['zero_installation'] = True
    controls[0]['zero_cache_writes'] = False
    assert decision(rows, controls, acceptance)['verdict'] == 'reject'
    controls[0]['zero_cache_writes'] = True
    controls[0]['zero_offpath_delta'] = False
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


def test_guard_rejects_desync_health_and_backwards_swap(tmp_path, monkeypatch):
    import time
    from types import SimpleNamespace

    from glm_rsi_dream import guard

    root = tmp_path / 'guard-run'
    root.mkdir()
    spec = {'original': 'original-example', 'candidates': ['candidate-example'],
            'deadline': time.time() + 3600, 'ready_after': 0,
            'api_url': 'http://example-head:8888',
            'safety': {'memory_floor_bytes': 1, 'swap_out_bytes_per_second': 10_000_000,
                       'swap_sustain_seconds': 30, 'probe_seconds': 5,
                       'dashboard_url': 'http://example-dashboard/status',
                       'protected_urls': ['http://example-service/health']}}
    path = root / 'guard.json'
    path.write_text(json.dumps(spec))

    class Response:
        status = 200

        def read(self, limit):
            return b'{"ok": false, "rank_desynchronized": true}'

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(guard, 'sample', lambda: (time.monotonic(), 10 ** 12, 0))
    monkeypatch.setattr(guard.urllib.request, 'urlopen', lambda *args, **kwargs: Response())
    monkeypatch.setattr(guard.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0))
    monkeypatch.setattr(guard.time, 'sleep', lambda seconds: (root / 'guard-done').write_text('done\n'))
    guard.run(path)
    fired = json.loads((root / 'guard-fired.json').read_text())
    assert fired['reason'] == 'protected health'

    samples = [(0.0, 10 ** 12, 100), (2.0, 10 ** 12, 40)]

    def sample():
        return samples.pop(0)

    def sleep(seconds):
        if not samples:
            (root / 'guard-done').write_text('done\n')

    (root / 'guard-fired.json').unlink()
    (root / 'guard-done').unlink(missing_ok=True)
    monkeypatch.setattr(guard, 'sample', sample)
    monkeypatch.setattr(guard.time, 'sleep', sleep)

    class Healthy:
        status = 200

        def read(self, limit):
            return b'ready'

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(guard.urllib.request, 'urlopen', lambda *args, **kwargs: Healthy())
    guard.run(path)
    fired = json.loads((root / 'guard-fired.json').read_text())
    assert fired['reason'] == 'memory telemetry is invalid'


def test_status_excludes_preserved_environment(tmp_path, capsys):
    write_json(tmp_path / 'state.json', {'phase': 'restored', 'originals': [{'Config': {'Env': ['TOKEN=private-example']}}]})
    assert main(['--run', str(tmp_path), 'status']) == 0
    assert 'private-example' not in capsys.readouterr().out
