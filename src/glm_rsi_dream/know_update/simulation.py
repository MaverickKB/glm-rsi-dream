"""Scripted CPU oracle used only to exercise evaluation software.

This is an explicit fixture oracle. Its answers are not GLM outputs and its
runtime snapshots are synthetic. Never interpret it as learning acceptance.
"""

import copy
from typing import Any

from .runner import cases_for, digest, step_spec, steps_for


class ScriptedBackend:
    def __init__(self, suite: Any, fault: Any = None) -> None:
        self.suite = suite
        self.fault = fault
        self.spec = step_spec(suite, steps_for('pocket')[0], digest([]))
        self.original_spec = copy.deepcopy(self.spec)
        self.epoch = 0
        self.transitions: list[Any] = []
        self.requests: list[Any] = []
        self.restored = False
        self.probes = 0

    def transition(self, spec: Any) -> Any:
        if spec['previous_facts_sha256'] != self.spec['facts_sha256']:
            raise ValueError('Synthetic predecessor differs')
        self.spec = copy.deepcopy(spec)
        self.epoch += 1
        self.transitions.append(copy.deepcopy(spec))
        return {'previous_facts_sha256': spec['previous_facts_sha256'],
                'facts_sha256': spec['facts_sha256'], 'synthetic': True}

    def inspect_runtime(self, probe_id: Any) -> Any:
        self.probes += 1
        ranks = []
        for rank_id in (0, 1):
            rank = {'tp_rank': rank_id, 'tp_world': 2, 'stale_cache_cleared': True,
                    'enabled_pockets': [], 'active_memory_rows': 0,
                    'generation_listener': rank_id == 0, 'runtime_identity_sha256': digest(['container', rank_id]),
                    'architecture': {'hidden_size': 4096, 'base_intermediate_size': 2048,
                                     'shared_experts': 1, 'routed_experts': 288, 'base_frozen': True,
                                     'shared_expert_additions': int(self.spec['neuron_facts_sha256'] is not None)}}
            for kind in ('pocket', 'neuron'):
                facts_hash = self.spec[kind + '_facts_sha256']
                rank[kind + '_enabled'] = facts_hash is not None
                rank[kind + '_facts_sha256'] = facts_hash
                rank[kind + '_sha256'] = None
                if facts_hash is not None:
                    rank[kind + '_sha256'] = digest([kind, rank_id, facts_hash])
            if rank['pocket_enabled']:
                rank['enabled_pockets'] = ['fictional-update']
                rank['active_memory_rows'] = 96
            ranks.append(rank)
        snapshot = {'probe_id': probe_id, 'epoch': str(self.epoch),
                    'model': 'GLM-5.3-Flash', 'generation_api_rank': 0,
                    'pair_sha256': 'a' * 64, 'base_sha256': 'b' * 64,
                    'facts_sha256': self.spec['facts_sha256'], 'ranks': ranks,
                    'synthetic': True}
        active = self.spec['phase'] != 'baseline' and not self.restored
        if self.fault == 'missing-rank' and active:
            snapshot['ranks'] = snapshot['ranks'][:1]
        if self.fault == 'stale-cache' and active:
            ranks[1]['stale_cache_cleared'] = False
        if self.fault == 'base-change' and active:
            snapshot['base_sha256'] = 'c' * 64
        if self.fault == 'worker-api' and active:
            ranks[1]['generation_listener'] = True
        if self.fault == 'pocket-left-on' and self.spec['phase'].startswith('dream-') and self.spec['pocket_facts_sha256'] is None:
            ranks[1]['pocket_enabled'] = True
        return snapshot

    def generate(self, request: Any) -> Any:
        self.requests.append(copy.deepcopy(request))
        question = request['messages'][0]['content']
        revision = None
        for index in range(3):
            facts = []
            for scenario in self.suite['scenarios']:
                facts.extend(scenario['revisions'][index]['facts'])
            if digest(facts) == self.spec['facts_sha256']:
                revision = index
        answer = 'No fictional fact has been learned.'
        case = None
        for candidate in cases_for(self.suite, revision):
            if candidate['prompt'] == question:
                case = candidate
                break
        if case is None:
            raise ValueError('Unknown scripted question')
        if case['kind'] == 'exact':
            answer = 'Scripted control ' + case['id']
        elif case['kind'] == 'leak':
            answer = 'No specific fictional equipment is known for this question.'
        elif revision is not None:
            answer = 'Current values: ' + ', '.join(case['expected']) + '.'
        active = revision is not None and not self.restored
        if self.fault == 'stale-answer' and revision in (1, 2) and case['kind'] == 'recall':
            answer = 'An older setting remains current.'
        if self.fault == 'mixed-answer' and revision in (1, 2) and case['kind'] == 'recall':
            answer += ' An older setting also remains current.'
        if self.fault == 'dependency-loss' and active and case['group'] in ('dependent', 'stable'):
            answer = 'I cannot recover that relationship.'
        if self.fault == 'control-drift' and active and case['kind'] == 'exact':
            answer += ' changed'
        if self.fault == 'leak' and active and case['kind'] == 'leak':
            answer += ' The Linden greenhouse relay uses private learned settings.'
        finish = 'stop'
        tokens = 12
        if self.fault == 'token-drift' and active and case['kind'] == 'exact':
            tokens = 13
        if self.fault == 'truncated' and active and case['kind'] == 'recall':
            finish = 'length'
        if self.fault == 'transport' and active:
            raise OSError('Scripted transport failure')
        return {'choices': [{'message': {'role': 'assistant', 'content': answer}, 'finish_reason': finish}],
                'usage': {'completion_tokens': tokens}, 'synthetic': True}

    def judge(self, case: Any, answer: Any) -> Any:
        if self.fault == 'invalid-grade':
            return {'correct': 'yes', 'reason': 'Malformed scripted grade'}
        if case['kind'] == 'leak':
            return {'leaked': 'private learned settings' in answer, 'reason': 'Scripted leakage oracle'}
        correct = True
        for value in case['expected']:
            if value not in answer:
                correct = False
        stale = 'older setting' in answer
        return {'correct': correct, 'grounded': correct and not stale,
                'stale': stale, 'reason': 'Scripted fixture oracle'}

    def rollback(self) -> Any:
        self.spec = copy.deepcopy(self.original_spec)
        self.epoch += 1
        self.restored = True
        if self.fault == 'rollback-failure':
            raise OSError('Scripted restore failure')
        return {'synthetic': True, 'restored': True,
                'owned_candidates_remaining': [], 'owned_trainers_remaining': []}
