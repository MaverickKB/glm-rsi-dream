import copy
import json
import socket
import subprocess
from pathlib import Path

import pytest

from glm_rsi_dream.know_update.__main__ import main, read_grant
from glm_rsi_dream.know_update.runner import digest, run, validate_suite
from glm_rsi_dream.know_update.simulation import ScriptedBackend

SUITE_PATH = Path(__file__).resolve().parents[2] / 'examples/know_update/suite.json'


@pytest.fixture
def suite():
    return json.loads(SUITE_PATH.read_text())


@pytest.mark.parametrize('arm,phases', [('pocket', 4), ('dream', 6)])
def test_full_trajectory_is_cpu_evidence_with_bare_requests(suite, arm, phases, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU validation launched a process or network operation')

    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    backend = ScriptedBackend(suite)
    events = []
    report = run(suite, backend, arm, 'cpu-simulation', events.append)
    assert report['passed']
    assert not report['native_acceptance']
    assert len(report['phases']) == phases
    assert report['rollback']['passed']
    for request in backend.requests:
        assert len(request['messages']) == 1
        assert request['messages'][0]['role'] == 'user'
        assert set(request['metadata']) == {'rsi_probe'}
        assert len(request['metadata']['rsi_probe']) == 32
        assert 'facts' not in request
        assert 'tools' not in request
    for transition in backend.transitions:
        assert 'cases' not in transition
        assert 'expected' not in transition
    sealed = copy.deepcopy(report)
    seal = sealed.pop('report_sha256')
    assert digest(sealed) == seal
    assert events[-1]['event'] == 'result'


@pytest.mark.parametrize('arm', ['pocket', 'dream'])
@pytest.mark.parametrize('fault', [
    'stale-answer', 'mixed-answer', 'dependency-loss', 'control-drift',
    'token-drift', 'leak', 'invalid-grade', 'truncated', 'transport',
    'missing-rank', 'stale-cache', 'base-change', 'worker-api', 'rollback-failure',
])
def test_failures_reject_acceptance_and_attempt_real_restore(suite, arm, fault):
    backend = ScriptedBackend(suite, fault)
    report = run(suite, backend, arm, 'cpu-simulation')
    assert not report['passed'], fault
    assert not report['native_acceptance']
    assert backend.restored


def test_postdream_recall_rejects_attention_left_enabled(suite):
    report = run(suite, ScriptedBackend(suite, 'pocket-left-on'), 'dream', 'cpu-simulation')
    assert not report['passed']
    assert 'Incomplete phase' in report['errors'][0]


def test_factual_failure_keeps_later_revisions_in_evidence(suite):
    report = run(suite, ScriptedBackend(suite, 'stale-answer'), 'pocket', 'cpu-simulation')
    assert len(report['phases']) == 4
    corrected = report['phases'][2]
    assert corrected['groups']['current']['passed'] == 0
    assert corrected['groups']['stale']['total'] == 3
    assert report['phases'][3]['phase'] == 'pocket-replaced'


def test_runtime_failure_stops_gpu_calls_without_shrinking_denominator(suite):
    backend = ScriptedBackend(suite, 'transport')
    report = run(suite, backend, 'pocket', 'cpu-simulation')
    failed = report['phases'][1]
    assert len(failed['records']) == 38
    assert failed['groups']['ordinary']['total'] == 12
    assert failed['groups']['protected']['total'] == 4
    assert sum('Not attempted' in row.get('error', '') for row in failed['records']) == 37
    assert len(backend.transitions) == 2
    assert report['rollback']['passed']


def test_restore_returns_original_live_carriers_not_test_baseline(suite):
    backend = ScriptedBackend(suite)
    original = {'phase': 'retained-service', 'facts_sha256': 'c' * 64,
                'pocket_facts_sha256': 'd' * 64, 'neuron_facts_sha256': None}
    backend.spec = copy.deepcopy(original)
    backend.original_spec = copy.deepcopy(original)
    report = run(suite, backend, 'pocket', 'cpu-simulation')
    assert report['passed']
    assert backend.spec == original
    assert report['rollback']['spec']['pocket_facts_sha256'] == 'd' * 64
    assert report['phases'][0]['spec']['pocket_facts_sha256'] is None


def test_inflight_state_change_is_rejected(suite):
    class DriftingBackend(ScriptedBackend):
        def generate(self, request):
            reply = super().generate(request)
            if self.spec['phase'] == 'pocket-original':
                self.epoch += 1
            return reply

    report = run(suite, DriftingBackend(suite), 'pocket', 'cpu-simulation')
    assert not report['passed']
    assert report['phases'][1]['records'][0]['error'] == 'Runtime changed during generation'


def test_one_shared_expert_architecture_is_enforced(suite):
    class WrongArchitecture(ScriptedBackend):
        def inspect_runtime(self, probe_id):
            state = super().inspect_runtime(probe_id)
            if self.spec['phase'] == 'dream-original':
                state['ranks'][1]['architecture']['routed_experts'] = 289
            return state

    report = run(suite, WrongArchitecture(suite), 'dream', 'cpu-simulation')
    assert not report['passed']
    assert 'shared-expert' in report['phases'][1]['records'][0]['error']


@pytest.mark.parametrize('mutation', ['broken-link', 'reused-id', 'answer-hint', 'extra-context', 'missing-stable'])
def test_fixture_shortcuts_are_rejected(suite, mutation):
    scenario = suite['scenarios'][0]
    if mutation == 'broken-link':
        scenario['revisions'][1]['facts'][1]['supersedes'] = ['unrelated.v1']
    elif mutation == 'reused-id':
        scenario['revisions'][1]['facts'][1]['id'] = scenario['revisions'][0]['facts'][1]['id']
    elif mutation == 'answer-hint':
        scenario['cases'][0]['prompt'] += ' coral.linden.test'
    elif mutation == 'extra-context':
        suite['parameters']['messages'] = [{'role': 'system', 'content': 'Facts'}]
    else:
        scenario['cases'] = [case for case in scenario['cases'] if case['group'] != 'stable']
    with pytest.raises(ValueError):
        validate_suite(suite)


def test_grant_check_precedes_adapter_loading_and_output_creation(suite, tmp_path):
    queue = tmp_path / 'GPU-QUEUE.md'
    queue.write_text('REQUEST only\n')
    adapter = tmp_path / 'adapter.py'
    marker = tmp_path / 'executed'
    adapter.write_text('from pathlib import Path\nPath(' + repr(str(marker)) + ').touch()\n')
    output = tmp_path / 'native-run'
    with pytest.raises(SystemExit):
        main(['native', '--suite', str(SUITE_PATH), '--queue', str(queue),
              '--adapter', str(adapter), '--output', str(output)])
    assert not marker.exists()
    assert not output.exists()


def test_grant_revocation_and_adapter_identity_are_checked(tmp_path):
    adapter = tmp_path / 'adapter.py'
    adapter.write_text('# approved source\n')
    import hashlib
    grant = {'job': 'KNOW-update', 'arm': 'pocket', 'expires': '2099-01-01T00:00:00-05:00',
             'generation': True, 'transitions': True,
             'adapter_sha256': hashlib.sha256(adapter.read_bytes()).hexdigest()}
    queue = tmp_path / 'GPU-QUEUE.md'
    queue.write_text('GRANT KNOW-update-pocket-01 ' + json.dumps(grant) + '\n')
    assert read_grant(queue, 'KNOW-update-pocket-01', 'pocket', adapter) == grant
    adapter.write_text('# changed source\n')
    with pytest.raises(ValueError, match='source differs'):
        read_grant(queue, 'KNOW-update-pocket-01', 'pocket', adapter)
    queue.write_text(queue.read_text() + 'REVOKE KNOW-update-pocket-01\n')
    with pytest.raises(ValueError, match='No explicit GPU grant'):
        read_grant(queue, 'KNOW-update-pocket-01', 'pocket', adapter)


def test_simulator_cannot_be_promoted_to_native_acceptance(suite):
    report = run(suite, ScriptedBackend(suite), 'pocket', 'native')
    assert not report['passed']
    assert not report['native_acceptance']
    assert 'Synthetic evidence' in report['errors'][-1]


def test_cycle_and_unsupported_expected_value_are_rejected(suite):
    cycle = copy.deepcopy(suite)
    cycle['scenarios'][0]['revisions'][0]['facts'][0]['depends_on'] = ['endpoint']
    with pytest.raises(ValueError, match='cycle'):
        validate_suite(cycle)
    suite['scenarios'][0]['revisions'][0]['facts'][0]['value'] = 'unsupported.test'
    with pytest.raises(ValueError, match='supported'):
        validate_suite(suite)


def test_rollback_orphan_is_not_hidden_by_exact_replies(suite):
    class OrphanBackend(ScriptedBackend):
        def rollback(self):
            receipt = super().rollback()
            receipt['owned_trainers_remaining'] = ['owned-test-trainer']
            return receipt

    report = run(suite, OrphanBackend(suite), 'pocket', 'cpu-simulation')
    assert not report['passed']
    assert 'owned_trainers_remaining' in report['errors'][-1]


def test_protected_tool_metadata_and_finish_changes_fail_exact_comparison(suite):
    class ExtraMetadata(ScriptedBackend):
        def generate(self, request):
            reply = super().generate(request)
            if self.spec['phase'] == 'pocket-corrected' and request['messages'][0]['content'] == suite['controls'][-1]['prompt']:
                reply['choices'][0]['message']['tool_calls'] = [{'id': 'unexpected'}]
            return reply

    report = run(suite, ExtraMetadata(suite), 'pocket', 'cpu-simulation')
    assert not report['passed']
    assert report['phases'][2]['groups']['protected']['passed'] == 3


def test_latest_revision_tracks_all_retired_versions(suite):
    backend = ScriptedBackend(suite)
    seen = []
    original_judge = backend.judge

    def judge(case, answer):
        if backend.spec['phase'] == 'pocket-replaced' and case['id'] == 'relay-2':
            seen.extend(case['retired_facts'])
        return original_judge(case, answer)

    backend.judge = judge
    assert run(suite, backend, 'pocket', 'cpu-simulation')['passed']
    retired_ports = []
    for fact in seen:
        if fact['key'] == 'port':
            retired_ports.append(fact['value'])
    assert retired_ports == ['7401', '8523']


def test_native_rejects_missing_freeze_before_any_effect(suite, tmp_path):
    custom = tmp_path / 'custom.json'
    custom.write_text(json.dumps(suite))
    with pytest.raises(SystemExit):
        main(['native', '--suite', str(custom)])


def test_native_rejects_frozen_question_change_before_any_effect(suite, tmp_path):
    custom = tmp_path / 'custom.json'
    custom.write_text(json.dumps(suite))
    freeze = custom.with_suffix('.freeze.json')
    freeze.write_text(json.dumps({'suite_sha256': digest(suite)}))
    suite['controls'][0]['prompt'] += ' Please.'
    custom.write_text(json.dumps(suite))
    with pytest.raises(SystemExit):
        main(['native', '--suite', str(custom)])


def test_original_runtime_cannot_change_before_control_capture(suite):
    class ChangedOriginal(ScriptedBackend):
        def inspect_runtime(self, probe_id):
            state = super().inspect_runtime(probe_id)
            if self.probes > 1 and not self.restored:
                state['ranks'][0]['runtime_identity_sha256'] = 'e' * 64
            return state

    report = run(suite, ChangedOriginal(suite), 'pocket', 'cpu-simulation')
    assert not report['passed']
    assert 'Original service changed' in report['errors'][0]
