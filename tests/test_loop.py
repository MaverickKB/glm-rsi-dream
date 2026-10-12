from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from glm_rsi_dream import loop
from glm_rsi_dream.config import load_config
from glm_rsi_dream.dataset import freeze
from glm_rsi_dream.files import read_json, write_json
from glm_rsi_dream.safety import SafetyError

ROOT = Path(__file__).resolve().parents[1]


class FakeCluster:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.config = load_config(ROOT / 'config.example.json')
        self.state = {'phase': 'new'}
        self.restores = 0

    @contextmanager
    def window(self, protected):
        self.state['phase'] = 'serving'
        try:
            yield []
        finally:
            self.restores += 1
            self.state['phase'] = 'restored'
            write_json(self.run_dir / 'restore-parity.json', {'byte_identical': len(protected)})


def test_loop_records_keep_only_after_restoration(tmp_path):
    cluster = FakeCluster(tmp_path / 'run')
    with patch('tokenizers.Tokenizer.from_file', return_value=object()), patch.object(loop, 'capture') as capture, \
         patch.object(loop, 'train', return_value=[{'path': 'rank0', 'sha256': 'a' * 64}]) as train, \
         patch.object(loop, 'check', return_value={'verdict': 'keep'}) as check:
        result = loop.run(cluster, ROOT / 'examples/synthetic/lesson.json', ROOT / 'examples/synthetic/eval.json')
    assert capture.call_count == train.call_count == check.call_count == 1
    assert cluster.restores == 1
    assert result['verdict'] == 'keep' and result['restored']
    assert read_json(cluster.run_dir / 'check.json')['restored']


def test_loop_rejects_and_restores_on_capture_or_grader_failure(tmp_path):
    cluster = FakeCluster(tmp_path / 'run')
    with patch('tokenizers.Tokenizer.from_file', return_value=object()), \
         patch.object(loop, 'capture', side_effect=SafetyError('bad teacher')), patch.object(loop, 'train') as train:
        with pytest.raises(SafetyError):
            loop.run(cluster, ROOT / 'examples/synthetic/lesson.json', ROOT / 'examples/synthetic/eval.json')
    assert cluster.restores == 1
    assert train.call_count == 0
    assert read_json(cluster.run_dir / 'check.json')['verdict'] == 'reject'


def test_check_marks_holdout_consumed_before_its_first_generation(tmp_path):
    cluster = FakeCluster(tmp_path / 'run')
    lesson, evaluation = freeze(cluster, ROOT / 'examples/synthetic/lesson.json', ROOT / 'examples/synthetic/eval.json')
    calls = []
    def evaluate(cluster, tokenizer, lesson, items, selected, label, compare=False):
        calls.append(label)
        if label == 'dev':
            return [{'memory': {'grounded': True}} for _ in items]
        assert (cluster.run_dir / 'evaluation-started.json').exists()
        raise SafetyError('generation failed after holdout opened')
    with patch.object(loop, 'evaluate_items', side_effect=evaluate):
        with pytest.raises(SafetyError):
            loop.check(cluster, object(), lesson, evaluation, [])
    assert calls == ['dev', 'heldout']
    with pytest.raises(SafetyError, match='fitting remains closed'):
        loop.train(cluster)


def test_failed_dev_leaves_holdout_unopened(tmp_path):
    cluster = FakeCluster(tmp_path / 'run')
    lesson, evaluation = freeze(cluster, ROOT / 'examples/synthetic/lesson.json', ROOT / 'examples/synthetic/eval.json')
    with patch.object(loop, 'evaluate_items', return_value=[{'memory': {'grounded': False}}]):
        result = loop.check(cluster, object(), lesson, evaluation, [])
    assert result['verdict'] == 'reject'
    assert not (cluster.run_dir / 'evaluation-started.json').exists()


def test_paired_receipts_require_exact_rank_artifact_and_zero_offpath(tmp_path):
    from glm_rsi_dream.controller import Cluster
    from glm_rsi_dream.native_cluster import AttentionCluster
    cluster = AttentionCluster(load_config(ROOT / 'config.example.json'), tmp_path / 'run')
    write_json(cluster.run_dir / 'lesson.json', read_json(ROOT / 'examples/synthetic/lesson.json'))
    write_json(cluster.run_dir / 'pair/prefix.json', {'teacher_ids': [1] * 128})
    tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens: '<|user|>Name the Linden project helper.<|assistant|>')
    body = {'prompt': [1, 2], 'metadata': {'dream_eval': {'arm': 'main'}}}
    selected = [{'sha256': 'a' * 64}, {'sha256': 'b' * 64}]
    receipts = []
    for item in selected:
        receipts.append({'memory_activated': True, 'memory_length': 96, 'cache_writes': 1, 'artifact_sha256': item['sha256']})
    with patch('tokenizers.Tokenizer.from_file', return_value=tokenizer), patch.object(Cluster, 'infer', return_value=({}, receipts)):
        cluster.infer(body, {'use_memory': True, 'selected_by_rank': selected})
        receipts[1]['artifact_sha256'] = 'a' * 64
        with pytest.raises(SafetyError, match='Rank artifact identity'):
            cluster.infer(body, {'use_memory': True, 'selected_by_rank': selected})
    receipts = [{'memory_activated': False, 'cache_writes': 1, 'offpath_delta': 0}] * 2
    with patch.object(Cluster, 'infer', return_value=({}, receipts)):
        with pytest.raises(SafetyError, match='Off-path'):
            cluster.infer(body, {'use_memory': False})


def test_protected_restore_failure_stays_visible(tmp_path):
    from glm_rsi_dream.controller import Cluster
    from glm_rsi_dream.native_cluster import AttentionCluster
    def failed_probe(*args, **kwargs):
        raise SafetyError('protected probe failed')
    transport = SimpleNamespace(call=failed_probe)
    cluster = AttentionCluster(load_config(ROOT / 'config.example.json'), tmp_path / 'run', transport)
    cluster.config['api_url'] = 'http://example-head:8888'
    cluster.state.update(phase='restored', restore_errors=[])
    with patch.object(Cluster, '_restore'):
        with pytest.raises(SafetyError):
            cluster._restore()
    assert cluster.state['phase'] == 'restore_failed'
    assert 'protected recovery' in cluster.state['restore_errors'][0]
