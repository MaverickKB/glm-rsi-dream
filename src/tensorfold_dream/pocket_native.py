# SPDX-License-Identifier: Apache-2.0
"""Attention-memory manifest, request policy and immutable receipts."""

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def load(path: Any,w: Any) -> Any:
    from glm_rsi_dream.files import digest, read_json
    path=Path(path)
    try:
        manifest=read_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError('Explicit attention-memory format required') from error
    if not isinstance(manifest, dict):
        raise ValueError('Explicit attention-memory format required')
    binding=manifest.get('binding')
    if not isinstance(binding, dict):
        raise ValueError('Explicit attention-memory format required')
    if manifest.get('schema_version')!=3 or binding.get('mechanism')!='attention_state_v1':
        raise ValueError('Explicit attention-memory format required')
    pair=manifest.get('pair_sha256')
    try:
        paired=digest(binding)==pair
    except (TypeError, ValueError):
        paired=False
    if not paired or w.world!=2 or binding.get('slots')!=96:
        raise ValueError('Paired manifest identity mismatch')
    prefix=path.parent/'prefix.json'
    prefix_sha=binding.get('prefix_sha256')
    try:
        actual=hashlib.sha256(prefix.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError('Frozen numerical prefix source mismatch') from error
    if not isinstance(prefix_sha, str) or actual!=prefix_sha:
        raise ValueError('Frozen numerical prefix source mismatch')
    return {'root':path.parent,'pair_sha256':manifest['pair_sha256'],
            'active':{},'receipts':Path('/z-receipts')}


def admitted(w: Any,record: Any,prompt: Any) -> Any:
    state=w.meta['tensorfold_dream.eval']
    path=state['root']/'requests'/(record['spec']['request_id']+'.json')
    policy=json.loads(path.read_text()) if path.exists() else {}
    record.update(prompt=list(prompt),policy=policy,captures={},artifact_sha256=None,
                  memory_activated=False,memory_length=0,cache_writes=0,offpath_delta=0.0)


def offpath_delta(record: Any) -> Any:
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


def finished(w: Any,record: Any) -> Any:
    packet_sha=None
    if record['captures']:
        path=Path('/capture')/(record['spec']['request_id']+'-rank'+str(w.rank)+'.pt')
        torch.save(record['captures'],path)
        packet_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    return {'mechanism':'attention_state_v1','memory_activated':record['memory_activated'],
            'memory_length':record['memory_length'],'artifact_sha256':record['artifact_sha256'],
            'capture_sha256':packet_sha,'cache_writes':int(record.get('cache_writes', 0)),
            'offpath_delta':offpath_delta(record)}
