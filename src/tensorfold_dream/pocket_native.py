# SPDX-License-Identifier: Apache-2.0
"""Attention-memory manifest, request policy and immutable receipts."""
import hashlib
import json
from pathlib import Path

import torch


def load(path,w):
    from glm_rsi_dream.files import digest
    path=Path(path)
    manifest=json.loads(path.read_text())
    binding=manifest['binding']
    if manifest['schema_version']!=3 or binding['mechanism']!='attention_state_v1':
        raise ValueError('Explicit attention-memory format required')
    if manifest['pair_sha256']!=digest(binding) or w.world!=2 or binding['slots']!=96:
        raise ValueError('Paired manifest identity mismatch')
    prefix=path.parent/'prefix.json'
    if hashlib.sha256(prefix.read_bytes()).hexdigest()!=binding['prefix_sha256']:
        raise ValueError('Frozen numerical prefix source mismatch')
    return {'root':path.parent,'pair_sha256':manifest['pair_sha256'],
            'active':{},'receipts':Path('/z-receipts')}


def admitted(w,record,prompt):
    state=w.meta['tensorfold_dream.eval']
    path=state['root']/'requests'/(record['spec']['request_id']+'.json')
    policy=json.loads(path.read_text()) if path.exists() else {}
    record.update(prompt=list(prompt),policy=policy,captures={},artifact_sha256=None,
                  memory_activated=False,memory_length=0,cache_writes=0,offpath_delta=0.0)


def offpath_delta(record):
    """Capture packets outside an attention capture, plus any stored meter.

    Cache writes are a different counter. This delta does not copy that counter
    or the activation flag.
    """
    stored = record.get('offpath_delta', 0.0)
    if stored is None:
        stored = 0.0
    extra = 0
    if not record.get('attention_capture'):
        captures = record.get('captures') or {}
        for parts in captures.values():
            extra += len(parts)
    return float(stored) + float(extra)


def finished(w,record):
    packet_sha=None
    if record['captures']:
        path=Path('/capture')/(record['spec']['request_id']+'-rank'+str(w.rank)+'.pt')
        torch.save(record['captures'],path)
        packet_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    return {'mechanism':'attention_state_v1','memory_activated':record['memory_activated'],
            'memory_length':record['memory_length'],'artifact_sha256':record['artifact_sha256'],
            'capture_sha256':packet_sha,'cache_writes':int(record.get('cache_writes', 0)),
            'offpath_delta':offpath_delta(record)}
