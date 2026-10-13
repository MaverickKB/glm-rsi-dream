"""One autonomous capture, fit, check, and keep/reject cycle on the Sparks."""

import json
import time
from typing import Any

from .dataset import close_fitting, freeze, verify
from .files import digest, read_json, sha, write_json
from .grading import assess, decision, text_of
from .manifest import MANIFEST
from .safety import SafetyError, require_parity


def event(cluster: Any, phase: Any, detail: Any) -> Any:
    path = cluster.run_dir / 'events.json'
    events = read_json(path) if path.exists() else []
    events.append({'time': time.time(), 'phase': phase, 'detail': detail})
    write_json(path, events)
    print(phase + ': ' + detail, flush=True)


def prompt_body(cluster: Any, tokenizer: Any, question: Any, main: Any = False) -> Any:
    prefix = read_json(cluster.run_dir / 'pair/prefix.json')
    suffix = tokenizer.encode('<|user|>' + question + '<|assistant|><think></think>', add_special_tokens=False).ids
    body = cluster.body(prefix['normal_ids'] + suffix, tagged=True,
                        arm='main' if main else 'none', completion=True)
    body['max_tokens'] = cluster.config['max_tokens']
    return body


def generated(cluster: Any, tokenizer: Any, task: Any, selected: Any = None, full: Any = False) -> Any:
    body = prompt_body(cluster, tokenizer, task['prompt'], main=full or selected is not None)
    policy = {'use_memory': full or selected is not None}
    if selected is not None:
        policy['selected_by_rank'] = selected
    reply, receipts = cluster.infer(body, policy)
    choice = reply['choices'][0]
    answer = choice.get('text', choice.get('message', {}).get('content', '')) or ''
    return {'answer': answer, 'complete': choice.get('finish_reason') == 'stop',
            'reply': reply, 'receipts': receipts}


def capture(cluster: Any, tokenizer: Any, lesson: Any) -> Any:
    if (cluster.run_dir / 'train-index.json').exists():
        raise SafetyError('Capture is immutable; start a new run')
    facts = []
    for fact in lesson['facts']:
        facts.append(fact['statement'])
    index = []
    teachers = []
    prefix = read_json(cluster.run_dir / 'pair/prefix.json')
    for task in lesson['training']:
        teacher = generated(cluster, tokenizer, task, full=True)
        review = assess(cluster, facts, task, teacher['answer'], teacher['complete'])
        teachers.append({'task': task, 'teacher': teacher, 'review': review})
        write_json(cluster.run_dir / 'teachers.json', teachers)
        if not review['grounded']:
            raise SafetyError('GLM teacher exceeded fact scope: ' + task['id'])
        body = prompt_body(cluster, tokenizer, task['prompt'], main=True)
        answer_start = len(prefix['teacher_ids']) + len(body['prompt']) - len(prefix['normal_ids']) - 1
        body['prompt'] += tokenizer.encode(teacher['answer'], add_special_tokens=False).ids
        body['max_tokens'] = 1
        _, receipts = cluster.infer(body, {'use_memory': True, 'attention_capture': True})
        request_id = body['metadata']['dream_eval']['request_id']
        for rank, receipt in enumerate(receipts):
            relative = 'capture/' + request_id + '-rank' + str(rank) + '.pt'
            raw = cluster.transport.read(rank, cluster.run_id, relative)
            import hashlib
            if hashlib.sha256(raw).hexdigest() != receipt.get('capture_sha256'):
                raise SafetyError('Teacher attention capture hash differs from receipt')
        index.append({'id': task['id'], 'request_id': request_id, 'answer_start': answer_start})
        write_json(cluster.run_dir / 'train-index.json', index)
    for rank in range(2):
        cluster.transport.write(rank, cluster.run_id, 'pair/train-index.json', json.dumps(index).encode())
    capture_identity = {'index_sha256': sha(cluster.run_dir / 'train-index.json'),
                        'teachers_sha256': sha(cluster.run_dir / 'teachers.json')}
    write_json(cluster.run_dir / 'capture-complete.json', capture_identity)
    event(cluster, 'capture', str(len(index)) + ' GLM-reviewed train-only attention trajectories')


def train(cluster: Any) -> Any:
    if (cluster.run_dir / 'evaluation-started.json').exists():
        raise SafetyError('Heldout opened; fitting remains closed')
    verify(cluster)
    sealed = read_json(cluster.run_dir / 'capture-complete.json')
    if sha(cluster.run_dir / 'train-index.json') != sealed['index_sha256']:
        raise SafetyError('Training index changed after capture')
    if sha(cluster.run_dir / 'teachers.json') != sealed['teachers_sha256']:
        raise SafetyError('Reviewed teachers changed after capture')
    selected = []
    for rank in range(2):
        container = cluster.state['candidates'][rank]['id']
        status = '/capture/trainer-status-rank' + str(rank) + '.json'
        command = ['python3', '-u', '-m', 'tensorfold_dream.fit_attention', '--rank', str(rank)]
        child = ('import subprocess,json\nfrom pathlib import Path\n'
                 'result=subprocess.run(' + repr(command) + ')\n'
                 'Path(' + repr(status) + ').write_text(json.dumps({"exit_code":result.returncode}))\n')
        code = ('import subprocess\nfrom pathlib import Path\n'
                'log=open("/capture/trainer-rank' + str(rank) + '.log","wb")\n'
                'p=subprocess.Popen(["python3","-u","-c",' + repr(child) + '],stdout=log,stderr=log)\n'
                'print(p.pid)')
        cluster.transport.call(rank, 'exec', container=container, code=code)
        deadline = time.monotonic() + cluster.config['safety']['window_seconds']
        while True:
            cluster.ensure_safe()
            code = ('from pathlib import Path\np=Path(' + repr(status) + ')\n'
                    'print(p.read_text() if p.exists() else "null")')
            result = json.loads(cluster.transport.call(rank, 'exec', container=container, code=code))
            if result is not None:
                if result['exit_code'] != 0:
                    raise SafetyError('Attention trainer failed on rank ' + str(rank))
                break
            if time.monotonic() >= deadline:
                raise SafetyError('Attention trainer exceeded recovery window')
            time.sleep(cluster.config['safety']['probe_seconds'])
        report = json.loads(cluster.transport.read(rank, cluster.run_id, 'capture/compiled-report-rank' + str(rank) + '.json'))
        raw = cluster.transport.read(rank, cluster.run_id, 'capture/compiled-rank' + str(rank) + '.pt')
        import hashlib
        if hashlib.sha256(raw).hexdigest() != report['sha256'] or report['rank'] != rank:
            raise SafetyError('Trained tensor identity differs from report')
        layers = []
        for item in report['layers']:
            layers.append(item['layer'])
        if layers != MANIFEST['memory']['layers']:
            raise SafetyError('Fit omitted a required DSA site')
        path = 'compiled-rank' + str(rank) + '.pt'
        cluster.transport.write(rank, cluster.run_id, 'pair/' + path, raw)
        local = cluster.run_dir / 'pair' / path
        local.write_bytes(raw)
        local.chmod(0o600)
        write_json(cluster.run_dir / ('fit-report-rank' + str(rank) + '.json'), report)
        selected.append({'path': path, 'sha256': report['sha256']})
    write_json(cluster.run_dir / 'selected.json', selected)
    event(cluster, 'train', '96 rows at all 11 DSA sites fitted sequentially on both ranks')
    return selected


def evaluate_items(cluster: Any, tokenizer: Any, lesson: Any, items: Any, selected: Any, label: Any, compare: Any = False) -> Any:
    facts = []
    for fact in lesson['facts']:
        facts.append(fact['statement'])
    rows = []
    for task in items:
        row = {'id': task['id'], 'fact_id': task['fact_id']}
        names = ['memory']
        if compare:
            names = ['baseline', 'fullcarrier', 'memory', 'off']
        for name in names:
            value = generated(cluster, tokenizer, task, selected if name == 'memory' else None,
                              full=name == 'fullcarrier')
            result = assess(cluster, facts, task, value['answer'], value['complete'])
            result.update(value)
            row[name] = result
        if compare:
            require_parity([row['baseline']['reply']], [row['off']['reply']], require_complete=False)
        rows.append(row)
        write_json(cluster.run_dir / (label + '.json'), rows)
    return rows


def control_receipts(receipts: Any) -> Any:
    """Installation, cache writes, and off-path delta are separate measurements."""
    activated = False
    cache_ok = True
    delta_ok = True
    for receipt in receipts:
        if receipt.get('memory_activated'):
            activated = True
        if receipt.get('cache_writes') != 0:
            cache_ok = False
        if receipt.get('offpath_delta') != 0:
            delta_ok = False
    return {'zero_installation': not activated,
            'zero_cache_writes': cache_ok,
            'zero_offpath_delta': delta_ok}


def check(cluster: Any, tokenizer: Any, lesson: Any, evaluation: Any, selected: Any) -> Any:
    dev = evaluate_items(cluster, tokenizer, lesson, lesson['development'], selected, 'dev')
    count = 0
    for row in dev:
        count += int(row['memory']['grounded'])
    if count / len(dev) < cluster.config['acceptance']['minimum_dev_recall']:
        return {'verdict': 'reject', 'reason': 'development grounded recall failed',
                'dev_grounded': count, 'dev_total': len(dev), 'heldout_opened': False}
    close_fitting(cluster, selected)
    rows = evaluate_items(cluster, tokenizer, lesson, evaluation['heldout'], selected, 'heldout', compare=True)
    controls = []
    for prompt in evaluation['protected'] + evaluation['near_miss']:
        task = {'prompt': prompt}
        baseline = generated(cluster, tokenizer, task)
        memory = generated(cluster, tokenizer, task, selected)
        require_parity([baseline['reply']], [memory['reply']], require_complete=False)
        effects = control_receipts(memory['receipts'])
        controls.append({'byte_identical': True, 'baseline': baseline, 'memory': memory,
                         'zero_installation': effects['zero_installation'],
                         'zero_cache_writes': effects['zero_cache_writes'],
                         'zero_offpath_delta': effects['zero_offpath_delta']})
        write_json(cluster.run_dir / 'controls.json', controls)
    result = decision(rows, controls, cluster.config['acceptance'])
    result.update(dev_grounded=count, dev_total=len(dev), heldout_opened=True)
    return result


def run(cluster: Any, lesson_path: Any, eval_path: Any) -> Any:
    lesson, evaluation = freeze(cluster, lesson_path, eval_path)
    result = None
    try:
        with cluster.window(evaluation['protected']):
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(cluster.run_dir / 'tokenizer.json'))
            capture(cluster, tokenizer, lesson)
            selected = train(cluster)
            result = check(cluster, tokenizer, lesson, evaluation, selected)
        if cluster.state['phase'] != 'restored' or not (cluster.run_dir / 'restore-parity.json').exists():
            raise SafetyError('Exact original restore acceptance is incomplete')
        result.update(selected=selected, restored=True, freeze_sha256=sha(cluster.run_dir / 'freeze.json'),
                      product_sha256=digest(MANIFEST))
        write_json(cluster.run_dir / 'check.json', result)
        event(cluster, result['verdict'], 'memory decision recorded; exact original pair restored')
        return result
    except BaseException as error:
        result = {'verdict': 'reject', 'reason': type(error).__name__ + ': ' + str(error),
                  'restored': cluster.state.get('phase') == 'restored',
                  'heldout_opened': (cluster.run_dir / 'evaluation-started.json').exists()}
        write_json(cluster.run_dir / 'check.json', result)
        raise


def ask(cluster: Any, question: Any) -> Any:
    verify(cluster)
    report = read_json(cluster.run_dir / 'check.json')
    if report['verdict'] != 'keep' or not report.get('restored'):
        raise SafetyError('Only a kept and restored memory can be read')
    if report['product_sha256'] != digest(MANIFEST) or report['freeze_sha256'] != sha(cluster.run_dir / 'freeze.json'):
        raise SafetyError('Kept memory contract changed')
    for item in report['selected']:
        if sha(cluster.run_dir / 'pair' / item['path']) != item['sha256']:
            raise SafetyError('Kept tensor identity changed')
    evaluation = read_json(cluster.run_dir / 'eval.json')
    with cluster.window(evaluation['protected']):
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(cluster.run_dir / 'tokenizer.json'))
        value = generated(cluster, tokenizer, {'prompt': question}, report['selected'])
        answer = text_of(value['reply'])
    return {'answer': answer, 'restored': cluster.state['phase'] == 'restored'}
