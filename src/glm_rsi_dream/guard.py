"""Independent host-local recovery if the head coordinator disappears."""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def sample():
    available = 0
    swapped = 0
    for line in Path('/proc/meminfo').read_text().splitlines():
        fields = line.split()
        if fields[0] == 'MemAvailable:':
            available = int(fields[1]) * 1024
    for line in Path('/proc/vmstat').read_text().splitlines():
        name, value = line.split()
        if name == 'pswpout':
            import os
            swapped = int(value) * os.sysconf('SC_PAGE_SIZE')
    return time.monotonic(), available, swapped


def run(path):
    root = Path(path).parent
    spec = json.loads(Path(path).read_text())
    policy = spec['safety']
    previous = None
    high_since = None
    reason = None
    while not (root / 'guard-done').exists():
        now, available, swapped = sample()
        if available < policy['memory_floor_bytes']:
            reason = 'memory floor'
        if time.time() >= spec['deadline']:
            reason = 'window recovery deadline'
        if previous is not None:
            elapsed = now - previous[0]
            rate = (swapped - previous[2]) / elapsed
            if rate > policy['swap_out_bytes_per_second']:
                if high_since is None:
                    high_since = previous[0]
                if now - high_since >= policy['swap_sustain_seconds']:
                    reason = 'sustained swap-out'
            else:
                high_since = None
        previous = now, available, swapped
        urls = [policy['dashboard_url']] + policy['protected_urls']
        if time.time() >= spec['ready_after']:
            urls.append(spec['api_url'] + '/health')
        try:
            for url in urls:
                with urllib.request.urlopen(url, timeout=5) as response:
                    if response.status != 200 or not response.read(65536):
                        raise RuntimeError('protected response failed')
        except Exception:
            reason = 'protected health'
        if reason is not None:
            errors = []
            for container in spec['candidates']:
                try:
                    result = subprocess.run(['docker', 'stop', '-t', '5', container], capture_output=True, timeout=40)
                    if result.returncode:
                        errors.append('candidate stop failed')
                except Exception:
                    errors.append('candidate stop timed out')
            try:
                result = subprocess.run(['docker', 'start', spec['original']], capture_output=True, timeout=40)
                if result.returncode:
                    errors.append('original start failed')
            except Exception:
                errors.append('original start timed out')
            (root / 'guard-fired.json').write_text(json.dumps({'reason': reason, 'errors': errors}))
            return
        time.sleep(policy['probe_seconds'])


if __name__ == '__main__':
    run(sys.argv[1])
