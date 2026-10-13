"""CPU verification and grant-bound native revision-suite entry point."""

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .runner import digest, run, validate_suite
from .simulation import ScriptedBackend


def read_grant(queue: Any, request_id: Any, arm: Any, adapter: Any) -> Any:
    prefix = 'GRANT ' + request_id + ' '
    grant = None
    for line in Path(queue).read_text().splitlines():
        if line.startswith('REVOKE ' + request_id):
            grant = None
        if line.startswith(prefix):
            grant = json.loads(line[len(prefix):])
    if grant is None:
        raise ValueError('No explicit GPU grant for ' + request_id)
    if grant.get('job') != 'KNOW-update' or grant.get('arm') != arm:
        raise ValueError('Grant job or arm differs')
    expires = datetime.fromisoformat(grant['expires'])
    if expires.tzinfo is None or expires <= datetime.now(timezone.utc):
        raise ValueError('Grant expired or lacks timezone')
    if grant.get('generation') is not True or grant.get('transitions') is not True:
        raise ValueError('Grant must authorize generation and lifecycle transitions')
    adapter_hash = hashlib.sha256(Path(adapter).read_bytes()).hexdigest()
    if grant.get('adapter_sha256') != adapter_hash:
        raise ValueError('Owner-approved adapter source differs')
    return grant


def load_backend(path: Any, suite: Any, grant: Any) -> Any:
    specification = importlib.util.spec_from_file_location('know_update_owner_adapter', path)
    if specification is None or specification.loader is None:
        raise ValueError('Cannot load owner adapter')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    needs_restart = getattr(module, 'REQUIRES_SERVING_RESTART', None)
    if type(needs_restart) is not bool:
        raise ValueError('Adapter must declare REQUIRES_SERVING_RESTART')
    if needs_restart and not grant.get('ken_restart_approval_ref'):
        raise ValueError('Serving restart requires owner approval')
    # The factory receives training facts and parameters, never evaluation cases.
    training = {'model_scope': suite['model_scope'], 'parameters': suite['parameters'], 'revisions': []}
    for index in range(3):
        facts = []
        for scenario in suite['scenarios']:
            facts.extend(scenario['revisions'][index]['facts'])
        training['revisions'].append(facts)
    return module.open_backend(training, grant)


class GrantedBackend:
    """Recheck grant expiry/revocation before every new GPU operation."""

    def __init__(self, backend: Any, queue: Any, request_id: Any, arm: Any, adapter: Any, controls: Any) -> None:
        self.backend = backend
        self.queue = queue
        self.request_id = request_id
        self.arm = arm
        self.adapter = adapter
        self.recovery = False
        self.controls = controls

    def authorize(self) -> Any:
        read_grant(self.queue, self.request_id, self.arm, self.adapter)

    def transition(self, spec: Any) -> Any:
        self.authorize()
        return self.backend.transition(spec)

    def generate(self, request: Any) -> Any:
        if self.recovery:
            messages = request.get('messages')
            if not isinstance(messages, list) or len(messages) != 1 or messages[0].get('content') not in self.controls:
                raise ValueError('Recovery generation is limited to original controls')
        else:
            self.authorize()
        return self.backend.generate(request)

    def judge(self, case: Any, answer: Any) -> Any:
        self.authorize()
        return self.backend.judge(case, answer)

    def inspect_runtime(self, probe_id: Any) -> Any:
        return self.backend.inspect_runtime(probe_id)

    def rollback(self) -> Any:
        # Recovery is still authorized when the time-limited work grant expires.
        self.recovery = True
        return self.backend.rollback()


def main(argv: Any = None) -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'simulate', 'native'))
    parser.add_argument('--suite', default='examples/know_update/suite.json')
    parser.add_argument('--arm', choices=('pocket', 'dream'), default='pocket')
    parser.add_argument('--output')
    parser.add_argument('--adapter')
    parser.add_argument('--queue', default='../../DREAM-neurons/GPU-QUEUE.md')
    parser.add_argument('--request-id', default='KNOW-update-pocket-01')
    args = parser.parse_args(argv)
    suite_path = Path(args.suite)
    suite = validate_suite(json.loads(suite_path.read_text()))
    freeze_path = suite_path.with_suffix('.freeze.json')
    if args.mode == 'native' or freeze_path.exists():
        try:
            freeze = json.loads(freeze_path.read_text())
        except (OSError, ValueError) as error:
            parser.error('Frozen evaluation identity unavailable: ' + str(error))
        if freeze.get('suite_sha256') != digest(suite):
            parser.error('Suite differs from its frozen evaluation identity')
    if args.mode == 'check':
        print(json.dumps({'valid': True, 'suite_sha256': digest(suite),
                          'scenarios': len(suite['scenarios']), 'gpu_operations': 0}))
        return 0
    backend = None
    if args.mode == 'native':
        if not args.adapter or not args.output:
            parser.error('native requires --adapter and an unused --output')
        try:
            grant = read_grant(args.queue, args.request_id, args.arm, args.adapter)
        except (ValueError, KeyError, OSError) as error:
            parser.error(str(error))
        # Check output admission before importing any native adapter.
    directory = None
    if args.output:
        directory = Path(args.output)
        directory.mkdir(parents=True, exist_ok=False)
    if args.mode == 'native':
        backend = load_backend(args.adapter, suite, grant)
        controls = []
        for case in suite['controls']:
            controls.append(case['prompt'])
        backend = GrantedBackend(backend, args.queue, args.request_id, args.arm, args.adapter, controls)
        evidence = 'native'
    else:
        backend = ScriptedBackend(suite)
        evidence = 'cpu-simulation'
    events = None
    if directory:
        (directory / 'suite.json').write_text(json.dumps(suite, indent=2) + '\n')
        events = (directory / 'events.jsonl').open('x')

    def emit(event: Any) -> Any:
        if events:
            events.write(json.dumps(event, ensure_ascii=False) + '\n')
            events.flush()

    try:
        report = run(suite, backend, args.arm, evidence, emit)
        if directory:
            temporary = directory / 'report.tmp'
            temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
            temporary.replace(directory / 'report.json')
    finally:
        if events:
            events.close()
    summary = {'evidence': evidence, 'arm': args.arm, 'passed': report['passed'],
               'native_acceptance': report['native_acceptance'], 'errors': report['errors'], 'phases': []}
    for phase in report['phases']:
        summary['phases'].append({'phase': phase['phase'], 'groups': phase.get('groups', {}),
                                  'passed': phase.get('passed', False)})
    print(json.dumps(summary, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
