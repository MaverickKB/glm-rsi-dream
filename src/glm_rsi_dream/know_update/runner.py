"""Revision evaluation with caller-owned lifecycle, generation and judgment.

No GPU, network, process or native runtime imports occur here. Facts go only to
transition and judge. Generation receives a bare question and an opaque audit ID.
"""

import copy
import hashlib
import json
import re
import uuid
from typing import Any


def digest(value: Any) -> Any:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def require_hash(value: Any, name: Any) -> Any:
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError('Missing verified hash: ' + name)


def revision_facts(suite: Any, revision: Any) -> Any:
    facts = []
    if revision is not None:
        for scenario in suite['scenarios']:
            for fact in scenario['revisions'][revision]['facts']:
                facts.append(copy.deepcopy(fact))
    return facts


def validate_suite(suite: Any) -> Any:
    if suite.get('schema_version') != 1 or suite.get('model_scope') != 'GLM-5.3-Flash':
        raise ValueError('Version 1 GLM-5.3-Flash suite required')
    parameters = suite.get('parameters', {})
    allowed = {'model', 'temperature', 'max_tokens', 'stream', 'draft', 'chat_template_kwargs'}
    if set(parameters) - allowed:
        raise ValueError('Parameters cannot add facts, tools or context')
    if parameters.get('temperature') != 0 or parameters.get('stream') is not False:
        raise ValueError('Deterministic complete replies required')
    if parameters.get('draft') is not False or parameters.get('chat_template_kwargs') != {'enable_thinking': False}:
        raise ValueError('Drafting and thinking must be disabled')
    if type(parameters.get('max_tokens')) is not int or parameters['max_tokens'] <= 0:
        raise ValueError('Positive max_tokens required')
    scenarios = suite.get('scenarios')
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError('Revision scenarios required')
    case_ids: set[Any] = set()
    scenario_ids = set()
    all_fact_ids = set()
    for scenario in scenarios:
        scenario_id = scenario['id']
        if not scenario_id or scenario_id in scenario_ids:
            raise ValueError('Unique scenario IDs required')
        scenario_ids.add(scenario_id)
        revisions = scenario['revisions']
        if len(revisions) != 3:
            raise ValueError('Original, correction and replacement required')
        previous: dict[Any, Any] = {}
        retired = set()
        for index, revision in enumerate(revisions):
            expected_kind = ('learn', 'correction', 'replacement')[index]
            if revision['kind'] != expected_kind:
                raise ValueError('Revision order changed')
            current = {}
            changed = 0
            for fact in revision['facts']:
                key = fact['key']
                if not key or key in current:
                    raise ValueError('Unique fact keys required')
                if not fact['statement'].strip() or not fact['value'].strip():
                    raise ValueError('Fact statement and value required')
                value_pattern = r'(?<!\w)' + re.escape(fact['value'].casefold()) + r'(?!\w)'
                if re.search(value_pattern, fact['statement'].casefold()) is None:
                    raise ValueError('Expected fact value must be supported by its statement')
                old = previous.get(key)
                if old is None:
                    if index != 0 or fact['supersedes']:
                        raise ValueError('Revision cannot silently introduce a fact')
                    if fact['id'] in all_fact_ids:
                        raise ValueError('Fact ID reused')
                    all_fact_ids.add(fact['id'])
                elif old != fact:
                    if fact['id'] == old['id'] or fact['id'] in all_fact_ids:
                        raise ValueError('Changed fact needs a fresh identity')
                    if fact['supersedes'] != [old['id']]:
                        raise ValueError('Changed fact must link its exact predecessor')
                    all_fact_ids.add(fact['id'])
                    retired.add(old['id'])
                    changed += 1
                if fact['id'] in retired:
                    raise ValueError('Retired fact cannot become active again')
                current[key] = fact
            if index and (set(current) != set(previous) or changed == 0):
                raise ValueError('Revision must preserve keys and change knowledge')
            for fact in current.values():
                for dependency in fact['depends_on']:
                    if dependency == fact['key'] or dependency not in current:
                        raise ValueError('Dependency must resolve to another active fact')
            resolved: set[Any] = set()
            while len(resolved) < len(current):
                progressed = False
                for key, fact in current.items():
                    if key in resolved:
                        continue
                    ready = True
                    for dependency in fact['depends_on']:
                        if dependency not in resolved:
                            ready = False
                    if ready:
                        resolved.add(key)
                        progressed = True
                if not progressed:
                    raise ValueError('Dependency cycle in the active catalog')
            previous = current
        groups = set()
        for case in scenario['cases']:
            groups.add(case['group'])
            if case['kind'] != 'recall' or case['group'] not in {'current', 'dependent', 'stable', 'stale'}:
                raise ValueError('Unknown recall group')
            for revision in revisions:
                facts_by_key = {}
                for fact in revision['facts']:
                    facts_by_key[fact['key']] = fact
                for key in case['fact_keys']:
                    fact = facts_by_key[key]
                    pattern = r'(?<!\w)' + re.escape(fact['value'].casefold()) + r'(?!\w)'
                    if re.search(pattern, case['prompt'].casefold()):
                        raise ValueError('Question reveals an answer: ' + case['id'])
            validate_case(case, case_ids)
        if groups != {'current', 'dependent', 'stable', 'stale'}:
            raise ValueError('Each scenario needs current, dependent, stable and stale probes')
    control_groups = set()
    for case in suite['controls']:
        if case['kind'] != 'exact' or case['group'] not in {'ordinary', 'protected'}:
            raise ValueError('Invalid control')
        control_groups.add(case['group'])
        validate_case(case, case_ids)
    if control_groups != {'ordinary', 'protected'} or not suite['leaks']:
        raise ValueError('Ordinary, protected and leakage probes required')
    for case in suite['leaks']:
        if case['kind'] != 'leak' or case['group'] != 'leak':
            raise ValueError('Invalid leakage probe')
        validate_case(case, case_ids)
    return suite


def validate_case(case: Any, identifiers: Any) -> Any:
    if not isinstance(case['id'], str) or not case['id'] or case['id'] in identifiers:
        raise ValueError('Unique case IDs required')
    identifiers.add(case['id'])
    if not isinstance(case['prompt'], str) or not case['prompt'].strip():
        raise ValueError('Bare prompt required')
    if case['kind'] == 'recall' and not case['fact_keys']:
        raise ValueError('Recall must target an active fact')


def cases_for(suite: Any, revision: Any) -> Any:
    cases = []
    target_revision = 0 if revision is None else revision
    for scenario in suite['scenarios']:
        active = scenario['revisions'][target_revision]['facts']
        by_key = {}
        for fact in active:
            by_key[fact['key']] = fact
        retired = []
        for older in scenario['revisions'][:target_revision]:
            for fact in older['facts']:
                if fact['value'] != by_key[fact['key']]['value'] and fact not in retired:
                    retired.append(fact)
        for source_case in scenario['cases']:
            case = copy.deepcopy(source_case)
            case['facts'] = copy.deepcopy(active)
            case['retired_facts'] = copy.deepcopy(retired)
            case['expected'] = []
            for key in case['fact_keys']:
                case['expected'].append(by_key[key]['value'])
            cases.append(case)
    cases.extend(copy.deepcopy(suite['controls']))
    for source_case in suite['leaks']:
        case = copy.deepcopy(source_case)
        case['facts'] = revision_facts(suite, target_revision)
        cases.append(case)
    return cases


def steps_for(arm: Any) -> Any:
    if arm == 'pocket':
        return [
            ('baseline', None, None, None, 'baseline'),
            ('pocket-original', 0, 0, None, 'learn'),
            ('pocket-corrected', 1, 1, None, 'correct'),
            ('pocket-replaced', 2, 2, None, 'replace'),
        ]
    if arm == 'dream':
        return [
            ('baseline', None, None, None, 'baseline'),
            ('dream-original', 0, None, 0, 'learn-and-dream'),
            ('dream-correction-staged', 1, 1, 0, 'correct-consolidated'),
            ('dream-corrected', 1, None, 1, 'redream'),
            ('dream-replacement-staged', 2, 2, 1, 'replace-consolidated'),
            ('dream-replaced', 2, None, 2, 'redream'),
        ]
    raise ValueError('Arm must be pocket or dream')


def step_spec(suite: Any, row: Any, previous_hash: Any) -> Any:
    name, revision, pocket_revision, neuron_revision, action = row
    facts = revision_facts(suite, revision)
    retired = []
    if revision is not None:
        active_ids = set()
        for fact in facts:
            active_ids.add(fact['id'])
        for older in range(revision):
            for fact in revision_facts(suite, older):
                if fact['id'] not in active_ids and fact['id'] not in retired:
                    retired.append(fact['id'])
    return {'phase': name, 'action': action, 'previous_facts_sha256': previous_hash,
            'facts': facts, 'facts_sha256': digest(facts), 'retired_ids': retired,
            'pocket_facts_sha256': None if pocket_revision is None else digest(revision_facts(suite, pocket_revision)),
            'neuron_facts_sha256': None if neuron_revision is None else digest(revision_facts(suite, neuron_revision))}


def validate_runtime(snapshot: Any, spec: Any, probe_id: Any) -> Any:
    if snapshot.get('probe_id') != probe_id or not snapshot.get('epoch'):
        raise ValueError('Runtime must bind the current probe and epoch')
    if snapshot.get('generation_api_rank') != 0:
        raise ValueError('Generation must be owned by the head')
    if snapshot.get('model') != 'GLM-5.3-Flash' or snapshot.get('facts_sha256') != spec['facts_sha256']:
        raise ValueError('Runtime model or effective fact catalog differs')
    for name in ('pair_sha256', 'base_sha256'):
        require_hash(snapshot.get(name), name)
    ranks = snapshot.get('ranks')
    if not isinstance(ranks, list) or len(ranks) != 2:
        raise ValueError('Both TP ranks required')
    seen = set()
    stable_ranks = []
    for rank in ranks:
        rank_id = rank.get('tp_rank')
        if type(rank_id) is not int or rank_id not in (0, 1) or rank_id in seen:
            raise ValueError('Distinct integer TP ranks required')
        seen.add(rank_id)
        if type(rank.get('tp_world')) is not int or rank['tp_world'] != 2:
            raise ValueError('TP=2 required')
        if rank_id == 1 and rank.get('generation_listener') is not False:
            raise ValueError('Worker must have no generation API')
        require_hash(rank.get('runtime_identity_sha256'), 'runtime identity')
        if rank.get('stale_cache_cleared') is not True:
            raise ValueError('Previous-revision KV and prefix caches must be cleared')
        selected = {'tp_rank': rank_id, 'tp_world': 2, 'stale_cache_cleared': True,
                    'runtime_identity_sha256': rank['runtime_identity_sha256'],
                    'generation_listener': rank['generation_listener']}
        for kind in ('pocket', 'neuron'):
            expected = spec[kind + '_facts_sha256']
            if rank.get(kind + '_enabled') is not (expected is not None):
                raise ValueError(kind + ' enabled state differs')
            if rank.get(kind + '_facts_sha256') != expected:
                raise ValueError(kind + ' catalog is stale or unbound')
            artifact = rank.get(kind + '_sha256')
            if expected is None:
                if artifact is not None:
                    raise ValueError('Disabled carrier must have no loaded artifact')
            else:
                require_hash(artifact, kind + ' artifact')
            selected[kind + '_sha256'] = artifact
            selected[kind + '_facts_sha256'] = expected
        selected['enabled_pockets'] = copy.deepcopy(rank.get('enabled_pockets'))
        selected['active_memory_rows'] = rank.get('active_memory_rows')
        if not isinstance(selected['enabled_pockets'], list) or type(selected['active_memory_rows']) is not int:
            raise ValueError('Loaded pocket names and row count required')
        if spec['pocket_facts_sha256'] is None:
            if rank.get('enabled_pockets') != [] or rank.get('active_memory_rows') != 0:
                raise ValueError('Every attention pocket and active row must be OFF')
        elif not selected['enabled_pockets'] or selected['active_memory_rows'] <= 0:
            raise ValueError('Enabled pocket must have actual loaded rows')
        layout = rank.get('architecture')
        required_layout = {'hidden_size': 4096, 'base_intermediate_size': 2048,
                           'shared_experts': 1, 'routed_experts': 288, 'base_frozen': True,
                           'shared_expert_additions': int(spec['neuron_facts_sha256'] is not None)}
        integer_fields = ('hidden_size', 'base_intermediate_size', 'shared_experts', 'routed_experts', 'shared_expert_additions')
        if not isinstance(layout, dict):
            raise ValueError('Actual model architecture required')
        for name in integer_fields:
            if type(layout.get(name)) is not int:
                raise ValueError('Architecture counts must be integers')
        if layout.get('base_frozen') is not True or layout != required_layout:
            raise ValueError('Frozen base and single shared-expert addition required')
        selected['architecture'] = copy.deepcopy(layout)
        stable_ranks.append(selected)
    stable_ranks.sort(key=lambda item: item['tp_rank'])
    return {'epoch': snapshot['epoch'], 'pair_sha256': snapshot['pair_sha256'],
            'base_sha256': snapshot['base_sha256'], 'facts_sha256': snapshot['facts_sha256'],
            'ranks': stable_ranks}


def reply_signature(reply: Any, exact: Any = False) -> Any:
    choices = reply.get('choices')
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError('One complete reply choice required')
    choice = choices[0]
    message = choice.get('message')
    if not isinstance(message, dict) or message.get('role') != 'assistant':
        raise ValueError('Assistant message required')
    content = message.get('content')
    if not isinstance(content, str) or not content.strip():
        raise ValueError('Nonempty answer required')
    if not exact and (message.get('tool_calls') or message.get('refusal')):
        raise ValueError('Factual generation must be a direct answer')
    finish = choice.get('finish_reason')
    if finish != 'stop' and not (exact and finish == 'length'):
        raise ValueError('Incomplete factual reply')
    tokens = reply.get('usage', {}).get('completion_tokens')
    if type(tokens) is not int or tokens <= 0:
        raise ValueError('Actual completion token count required')
    return {'message': copy.deepcopy(message), 'finish_reason': finish,
            'completion_tokens': tokens, 'token_sha256': reply.get('tensorfold', {}).get('token_sha')}


def verdict_pass(case: Any, verdict: Any) -> Any:
    if not isinstance(verdict, dict) or not isinstance(verdict.get('reason'), str) or not verdict['reason'].strip():
        raise ValueError('Explicit semantic reason required')
    flags = ('leaked',) if case['kind'] == 'leak' else ('correct', 'grounded', 'stale')
    for flag in flags:
        if type(verdict.get(flag)) is not bool:
            raise ValueError('Semantic flags must be booleans: ' + flag)
    if case['kind'] == 'leak':
        return not verdict['leaked']
    return verdict['correct'] and verdict['grounded'] and not verdict['stale']


def run_phase(suite: Any, backend: Any, spec: Any, revision: Any, baseline: Any, identity: Any, emit: Any) -> Any:
    records = []
    cases = cases_for(suite, revision)
    if spec['phase'] in ('original-service', 'rollback'):
        cases = copy.deepcopy(suite['controls'])
    stopped = False
    phase_state = None
    for case in cases:
        row = {'id': case['id'], 'kind': case['kind'], 'group': case['group'], 'passed': False}
        records.append(row)
        if stopped:
            row['error'] = 'Not attempted after runtime/generation failure'
            continue
        probe_id = uuid.uuid4().hex
        request = copy.deepcopy(suite['parameters'])
        request['messages'] = [{'role': 'user', 'content': case['prompt']}]
        request['metadata'] = {'rsi_probe': probe_id}
        row['request'] = request
        try:
            row['before'] = copy.deepcopy(backend.inspect_runtime(probe_id))
            before = validate_runtime(row['before'], spec, probe_id)
            pair_identity = (before['pair_sha256'], before['base_sha256'])
            if identity and pair_identity != identity:
                raise ValueError('Frozen base or pair changed across revisions')
            if phase_state is not None and before != phase_state:
                raise ValueError('Runtime changed during the phase')
            phase_state = before
            row['reply'] = copy.deepcopy(backend.generate(copy.deepcopy(request)))
            signature = reply_signature(row['reply'], exact=case['kind'] == 'exact' and case['group'] == 'ordinary')
            row['signature'] = signature
            row['after'] = copy.deepcopy(backend.inspect_runtime(probe_id))
            after = validate_runtime(row['after'], spec, probe_id)
            if before != after:
                raise ValueError('Runtime changed during generation')
        except Exception as error:
            row['error'] = str(error)
            stopped = True
            emit({'event': 'case', 'phase': spec['phase'], 'record': row})
            continue
        try:
            if case['kind'] == 'exact':
                if baseline is None:
                    row['passed'] = True
                else:
                    row['passed'] = signature == baseline[case['id']]
                    row['comparison'] = 'complete message, bytes, finish, completion tokens and optional token hash'
            else:
                row['judgment'] = copy.deepcopy(backend.judge(copy.deepcopy(case), signature['message']['content']))
                row['passed'] = verdict_pass(case, row['judgment'])
        except Exception as error:
            row['error'] = str(error)
        emit({'event': 'case', 'phase': spec['phase'], 'record': row})
    groups: dict[Any, Any] = {}
    for row in records:
        counts = groups.setdefault(row['group'], {'total': 0, 'passed': 0, 'errors': 0})
        counts['total'] += 1
        counts['passed'] += int(row['passed'])
        counts['errors'] += int('error' in row)
    complete = True
    passed = True
    for row in records:
        if 'error' in row:
            complete = False
        if not row['passed']:
            passed = False
    result = {'phase': spec['phase'], 'spec': spec, 'records': records, 'groups': groups,
              'complete': complete, 'passed': passed, 'runtime_verified': not stopped}
    emit({'event': 'phase', 'phase': spec['phase'], 'complete': complete, 'passed': passed})
    return result, phase_state


def run(suite: Any, backend: Any, arm: Any = 'pocket', evidence: Any = 'native', emit: Any = None) -> Any:
    """Execute one fresh trajectory, preserving failures and always restoring.

    Native authority is checked by the CLI before loading an adapter. The adapter
    owns transition, inspect_runtime, generate, judge, rollback and cancellation.
    emit streams evidence so interruption cannot erase all earlier observations.
    """
    suite = copy.deepcopy(validate_suite(suite))
    steps = steps_for(arm)
    if evidence not in ('native', 'cpu-simulation'):
        raise ValueError('Unknown evidence domain')
    if emit is None:
        def emit(event: Any) -> Any:
            pass
    report = {'schema_version': 1, 'suite_sha256': digest(suite), 'arm': arm,
              'evidence': evidence, 'phases': [], 'errors': [], 'passed': False,
              'native_acceptance': False}
    baseline = None
    identity = None
    original_artifacts = None
    previous_hash = digest([])
    original_spec = None
    try:
        probe_id = uuid.uuid4().hex
        original = copy.deepcopy(backend.inspect_runtime(probe_id))
        require_hash(original.get('facts_sha256'), 'original effective catalog')
        first_rank = original['ranks'][0]
        original_spec = {'phase': 'original-service', 'facts_sha256': original['facts_sha256'],
                         'pocket_facts_sha256': first_rank['pocket_facts_sha256'],
                         'neuron_facts_sha256': first_rank['neuron_facts_sha256']}
        original_state = validate_runtime(original, original_spec, probe_id)
        identity = (original_state['pair_sha256'], original_state['base_sha256'])
        original_artifacts = original_state['ranks']
        report['original_snapshot'] = original
        report['original_service'], service_state = run_phase(suite, backend, original_spec, None, None, identity, emit)
        if service_state != original_state:
            raise ValueError('Original service changed before control capture')
        if not report['original_service']['passed']:
            raise ValueError('Original control capture incomplete')
        baseline = {}
        for record in report['original_service']['records']:
            baseline[record['id']] = record['signature']
        report['original_snapshot'] = original
        previous_hash = original_spec['facts_sha256']
        for row in steps:
            spec = step_spec(suite, row, previous_hash)
            receipt = copy.deepcopy(backend.transition(copy.deepcopy(spec)))
            report['phases'].append({'phase': spec['phase'], 'transition': receipt})
            emit({'event': 'transition', 'phase': spec['phase'], 'spec': spec, 'receipt': receipt})
            if receipt.get('previous_facts_sha256') != previous_hash or receipt.get('facts_sha256') != spec['facts_sha256']:
                raise ValueError('Transition receipt does not bind both revision catalogs')
            phase, state = run_phase(suite, backend, spec, row[1], baseline, identity, emit)
            phase['transition'] = receipt
            report['phases'][-1] = phase
            if not phase['complete'] or not phase['runtime_verified']:
                raise ValueError('Incomplete phase: ' + spec['phase'])
            if spec['phase'] == 'baseline':
                for record in phase['records']:
                    if record['kind'] == 'exact' and not record['passed']:
                        raise ValueError('Baseline changed an original service control')
            previous_hash = spec['facts_sha256']
    except Exception as error:
        report['errors'].append(str(error))
    finally:
        try:
            report['restore_receipt'] = copy.deepcopy(backend.rollback())
            if original_spec is None:
                raise ValueError('Original runtime was not captured')
            for name in ('owned_candidates_remaining', 'owned_trainers_remaining'):
                if report['restore_receipt'].get(name) != []:
                    raise ValueError('Rollback must enumerate owned residuals: ' + name)
            rollback_spec = copy.deepcopy(original_spec)
            rollback_spec['phase'] = 'rollback'
            rollback, restored = run_phase(suite, backend, rollback_spec, None, baseline, identity, emit)
            if baseline is None or restored is None or restored['ranks'] != original_artifacts:
                raise ValueError('Original carrier state and baseline were not proven restored')
            report['rollback'] = rollback
        except Exception as error:
            report['errors'].append('Rollback: ' + str(error))
    passed = not report['errors'] and len(report['phases']) == len(steps)
    for phase in report['phases'][1:]:
        passed = passed and phase.get('passed', False)
    passed = passed and report.get('rollback', {}).get('passed', False)
    report['passed'] = bool(passed)
    synthetic = False
    for phase in report['phases']:
        if phase.get('transition', {}).get('synthetic') is True:
            synthetic = True
        for record in phase.get('records', []):
            for name in ('reply', 'before', 'after'):
                if record.get(name, {}).get('synthetic') is True:
                    synthetic = True
    report['native_acceptance'] = bool(passed and evidence == 'native' and not synthetic)
    if synthetic and evidence == 'native':
        report['errors'].append('Synthetic evidence cannot establish native acceptance')
        report['passed'] = False
    report['report_sha256'] = digest(report)
    emit({'event': 'result', 'passed': report['passed'], 'native_acceptance': report['native_acceptance']})
    return report
