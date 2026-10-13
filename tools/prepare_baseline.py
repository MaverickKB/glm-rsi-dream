"""Print or execute fresh-baseline provisioning from config, on the head Spark.

Existing container names always stop this operation before any creation. The
pocket controller owns every later clone/cutover/restore of these originals.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from glm_rsi_dream.config import load_config, require_spark, shell_command
from glm_rsi_dream.safety import MemoryGate
from glm_rsi_dream.transport import Transport


def plans(config: Any) -> Any:
    runtime = config['runtime']
    baseline = config['baseline']
    result = []
    for host in config['hosts']:
        rank = host['rank']
        root = host['work_dir'] + '/baseline'
        download = 'from huggingface_hub import snapshot_download\n'
        for role in ('model', 'drafter'):
            target = 'main' if role == 'model' else 'drafter'
            download += 'snapshot_download(repo_id=' + repr(runtime[role + '_repository'])
            download += ', revision=' + repr(runtime[role + '_revision'])
            download += ', local_dir=' + repr('/models/' + target)
            download += ', allow_patterns=' + repr(['model-*.safetensors', 'model.safetensors', 'model.safetensors.index.json', 'config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json', 'processor_config.json', 'quantization_config.json', 'exl3-mcg-storage-abi.json', 'LICENSE', 'THIRD_PARTY_NOTICES.md']) + ')\n'
        steps = []
        steps.append(['mkdir', '-p', root])
        steps.append(['docker', 'pull', runtime['image']])
        steps.append(['docker', 'run', '--rm', '--entrypoint', 'python3', '-v', root + ':/models',
                      runtime['image'], '-c', download])
        command = ['docker', 'create', '--name', host['original_container'], '--gpus', 'all',
                   '--network', 'host', '--ipc', 'host', '--cap-add', 'IPC_LOCK',
                   '--ulimit', 'memlock=-1:-1', '--ulimit', 'stack=67108864:67108864',
                   '--restart', 'no', '-v', root + ':/models:ro']
        for key, value in baseline['environment'].items():
            command.extend(['-e', key + '=' + str(value)])
        command.extend([runtime['image'], 'tensorfold', 'serve', '/models/main',
                        '--tp', '2', '--rank', str(rank), '--master', baseline['master_address'],
                        '--master-port', str(baseline['master_port']), '--drafter', '/models/drafter',
                        '--context', str(baseline['context_tokens']), '--parallel', '4',
                        '--max-tokens', str(baseline['maximum_tokens']), '--thinking', '--vision'])
        if rank == 0:
            command.extend(['--name', config['model'], '--host', baseline['api_bind'],
                            '--port', str(baseline['api_port'])])
        steps.append(command)
        result.append({'ssh': host['ssh'], 'container': host['original_container'], 'steps': steps})
    return result


def remote(config: Any, target: Any, command: Any, check: Any = True) -> Any:
    # A single shell command is required by SSH's remote command protocol.
    # A leading ~/ stays expandable as $HOME on that Spark.
    return subprocess.run(config['ssh_command'] + [target, shell_command(command)], check=check)


def main(argv: Any=None) -> Any:
    parser = argparse.ArgumentParser(description='Fresh baseline plan; default is offline dry-run')
    parser.add_argument('--config', default='config.json',
                        help='config file (default: config.json)')
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args(argv)
    config = load_config(args.config, live=args.live)
    plan = plans(config)
    if not args.live:
        print(json.dumps({'mode': 'dry-run', 'plans': plan, 'start_order': [1, 0]}, indent=2))
        return 0
    require_spark()
    if '<' in json.dumps(config):
        raise ValueError('replace all configuration placeholders before provisioning')
    # Inspect both names before pulling, downloading, creating or starting anything.
    for host in plan:
        probe = remote(config, host['ssh'], ['docker', 'container', 'inspect', host['container']], check=False)
        if probe.returncode == 0:
            raise ValueError('baseline container already exists; run the learning loop against it')
        if probe.returncode != 1:
            raise ValueError('container preflight could not establish absence')
        remote(config, host['ssh'], ['docker', 'info'])
    for host in plan:
        for command in host['steps']:
            remote(config, host['ssh'], command)
    transport = Transport(config)
    gate = MemoryGate(config['safety'])
    for rank in range(2):
        gate.observe(rank, transport.call(rank, 'metrics'))
    started = []
    try:
        for rank in (1, 0):
            remote(config, plan[rank]['ssh'], ['docker', 'start', plan[rank]['container']])
            started.append(rank)
        deadline = time.monotonic() + config['safety']['ready_timeout_seconds']
        while True:
            for rank in range(2):
                gate.observe(rank, transport.call(rank, 'metrics'))
                record = transport.call(rank, 'inspect', container=plan[rank]['container'])
                if not record['State']['Running']:
                    raise RuntimeError('fresh baseline rank exited during initialization')
            ready = False
            try:
                with urllib.request.urlopen(config['api_url'] + '/health', timeout=5) as response:
                    ready = response.status == 200 and bool(response.read(65536))
            except (OSError, urllib.error.URLError):
                pass
            if ready:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('fresh baseline readiness exceeded the bounded startup window')
            time.sleep(config['safety']['probe_seconds'])
    except BaseException:
        for rank in (0, 1):
            remote(config, plan[rank]['ssh'], ['docker', 'stop', plan[rank]['container']], check=False)
        raise
    print('Baseline started. Verify health and a real reply, then run the learning loop.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
