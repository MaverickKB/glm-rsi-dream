"""Observable evaluation state for a paired pocket-off native serving epoch."""

import json
import os
from pathlib import Path
from typing import Any

KEY = 'tensorfold_dream.eval'


def record_clean_start(w: Any, epoch: Any) -> Any:
    """Called only after the native startup reset and pill bypass execute."""
    w.meta['dream_neurons_clean_start'] = {
        'epoch': epoch, 'engine_pid': os.getpid(),
        'reset': True, 'attention_startup_bypassed': True}


def attention_disabled(w: Any) -> Any:
    state = w.meta.get(KEY)
    if state is None:
        return False
    path = state['root'] / 'attention-disabled.json'
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    if value.get('all_attention_disabled') is not True or not value.get('owner'):
        raise ValueError('Invalid attention evaluation gate')
    return True


def neurons_enabled(w: Any) -> Any:
    state = w.meta.get(KEY)
    return state is not None and (state['root'] / 'neurons-current.json').exists()


def status(w: Any) -> Any:
    state = w.meta[KEY]
    if not attention_disabled(w):
        return None
    current = state['root'] / 'neurons-current.json'
    if current.exists():
        from .native import artifact

        selected = json.loads(current.read_text())['by_rank'][w.rank]
        artifact(w, selected)
    cache = w.meta.get('dream_neurons', {})
    active_rows = 0
    for record in state['active'].values():
        if record.get('memory_activated'):
            active_rows += record.get('memory_length', 0)
    if (state.get('snapshot') is not None or state.get('memory_snapshots')
            or state.get('compressed') is not None or active_rows):
        raise ValueError('Pocket-off serving epoch contains attention memory state')
    automatic = json.loads((state['root'] / 'automatic.json').read_text())
    if automatic.get('status') == 'enabled':
        raise ValueError('Automatic attention pockets remain enabled')
    clean = w.meta.get('dream_neurons_clean_start')
    flag = json.loads((state['root'] / 'attention-disabled.json').read_text())
    if (not clean or clean['engine_pid'] != os.getpid()
            or clean['epoch'] != flag.get('epoch')):
        raise ValueError('Pocket-off startup reset has no current engine/epoch receipt')
    value = {'tp_rank': w.rank, 'tp_world': w.world, 'engine_pid': os.getpid(),
             'clean_start': clean,
             'pair_sha256': state['pair_sha256'], 'pockets_enabled': False,
             'enabled_pockets': [], 'active_memory_rows': active_rows,
             'pocket_cache_clean': True, 'neurons_enabled': current.exists(),
             'neurons_sha256': cache.get('sha256') if current.exists() else None}
    path = Path('/capture') / ('runtime-rank' + str(w.rank) + '.json')
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True) + '\n')
    temporary.replace(path)
    return value
