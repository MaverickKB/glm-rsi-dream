"""Validate host-owned settings before any live operation."""

import json
import os
import platform
import shlex
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .files import read_json
from .manifest import MANIFEST

# One data directory for the coordinator run and each Spark's work_dir.
# "~" stays in the config so each host expands it for its own account.
DEFAULT_DATA_DIR = "~/.cache/glm-rsi-dream"
DEFAULT_RUN_DIR = DEFAULT_DATA_DIR + "/demo-1"


def expand_home(path: Any) -> Any:
    return os.path.expanduser(path)


def acceptable_work_dir(path: Any) -> Any:
    if not isinstance(path, str) or not path or path != path.strip():
        return False
    return expand_home(path).startswith("/")


def shell_command(command: Any) -> Any:
    """Quote a remote command, leaving a leading ~/ for the remote shell."""
    parts = []
    for arg in command:
        if arg.startswith("~/"):
            parts.append("$HOME/" + shlex.quote(arg[2:]))
        else:
            parts.append(shlex.quote(arg))
    return " ".join(parts)


def require_spark() -> Any:
    if platform.system() != 'Linux' or platform.machine() not in ('aarch64', 'arm64'):
        raise ValueError('Live work runs on the head Spark (Linux ARM64)')
    if not Path('/etc/dgx-release').is_file():
        raise ValueError('Live coordinator requires a DGX Spark host')


def load_config(path: Any, live: Any=False) -> Any:
    config = read_json(path)
    if live and '<' in json.dumps(config):
        raise ValueError('Replace configuration placeholders before live work')
    hosts = config['hosts']
    if len(hosts) != 2:
        raise ValueError('Exactly two Spark ranks are required')
    for rank, host in enumerate(hosts):
        if host['rank'] != rank or not host['ssh'] or not host['original_container']:
            raise ValueError('Host entries must own ranks 0 and 1 in order')
        if not acceptable_work_dir(host['work_dir']):
            raise ValueError('Work directory must be absolute on each Spark')
    if hosts[0]['ssh'] == hosts[1]['ssh']:
        raise ValueError('Two distinct Spark SSH targets are required')
    safety = config['safety']
    bounds = {'memory_floor_bytes': (4_000_000_000, None),
              'swap_out_bytes_per_second': (1, 10_000_000),
              'swap_sustain_seconds': (1, 30), 'probe_seconds': (1, 5),
              'ready_timeout_seconds': (1, 600), 'window_seconds': (60, 86400)}
    for name, (minimum, maximum) in bounds.items():
        value = safety[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError('Invalid safety setting: ' + name)
        if maximum is not None and value > maximum:
            raise ValueError('Safety bound exceeded: ' + name)
    if not safety['protected_urls']:
        raise ValueError('Configure protected service probes on both hosts')
    for url in [config['api_url'], safety['dashboard_url']] + safety['protected_urls']:
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname:
            raise ValueError('Invalid probe URL')
    for name in ('ssh_command', 'remote_python'):
        values = config[name]
        if not isinstance(values, list) or not values:
            raise ValueError('Commands must be nonempty argument lists')
        for value in values:
            if not isinstance(value, str) or not value:
                raise ValueError('Command arguments must be nonempty strings')
    for name in ('max_tokens', 'capture_max_tokens'):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 32768:
            raise ValueError('Invalid token bound: ' + name)
    for name in ('minimum_grounded_recall', 'minimum_gain', 'minimum_dev_recall'):
        value = config['acceptance'][name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
            raise ValueError('Invalid acceptance threshold: ' + name)
    if config['training']['device'] != 'cuda' or config['training']['threads'] != 4:
        raise ValueError('Native trainer requires CUDA with four CPU threads')
    config['runtime'] = MANIFEST['runtime']
    config['model'] = MANIFEST['model']
    return config
