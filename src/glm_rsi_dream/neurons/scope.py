"""Verify training membership and immutable study inputs before native work."""

import hashlib
import json
from pathlib import Path
from typing import Any

from ..files import digest


def verify_scope(root: Any) -> Any:
    root = Path(root)
    freeze = json.loads((root / 'study-freeze.json').read_text())
    declared = freeze.pop('sha256')
    if declared != digest(freeze):
        raise ValueError('Study freeze seal or role differs')
    if freeze.get('before_capture_and_fitting') is not True:
        origin = freeze.get('inherited_capture', {})
        if (freeze.get('before_fitting') is not True
                or len(origin.get('original_index_sha256', '')) != 64
                or origin.get('membership_audited') is not True):
            raise ValueError('Reused captures require an explicit audited source binding before fitting')
        original = root / 'capture-index-original.json'
        if hashlib.sha256(original.read_bytes()).hexdigest() != origin['original_index_sha256']:
            raise ValueError('Original inherited capture index changed')
        previous = json.loads((root / 'study-freeze-original.json').read_text())
        previous_sha = previous.pop('sha256')
        if previous_sha != digest(previous) or previous_sha != origin['original_study_sha256']:
            raise ValueError('Original inherited study seal changed')
        source_root = (root / origin.get('source_files_path', '.')).resolve()
        if not source_root.is_relative_to(root.resolve()):
            raise ValueError('Inherited capture source escaped its root')
        for name, expected_sha in previous['files'].items():
            path = (source_root / name).resolve()
            if not path.is_relative_to(source_root):
                raise ValueError('Inherited capture input escaped its source')
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
                raise ValueError('Inherited capture input changed: ' + name)
    required = ('train.json', 'dev.json', 'fresh.json', 'comparison.json',
                'pockets.json', 'ordinary-source.json', 'protected.json')
    for name in required:
        if name not in freeze['files']:
            raise ValueError('Study omitted required input: ' + name)
    for name, expected_sha in freeze['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Study input escaped its root')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha != actual:
            raise ValueError('Study input differs from its freeze: ' + name)
    expected = {}
    for item in json.loads((root / 'train.json').read_text()):
        if item['positive'] is not True or item['id'] in expected:
            raise ValueError('Positive training membership differs')
        expected[item['id']] = True
    for number, _ in enumerate(json.loads((root / 'ordinary-source.json').read_text())[:12]):
        expected['negative-' + str(number)] = False
    for name in ('dev.json', 'fresh.json', 'comparison.json'):
        for item in json.loads((root / name).read_text()):
            if item['id'] in expected:
                raise ValueError('Reserved example entered fitting')
    return declared, expected


def verify_examples(examples: Any, expected: Any, complete: Any = True) -> Any:
    seen = set()
    for item in examples:
        if item['id'] in seen or type(item['positive']) is not bool:
            raise ValueError('Duplicate example or invalid scope label')
        if item['id'] not in expected or item['positive'] != expected[item['id']]:
            raise ValueError('Capture example differs from frozen training membership')
        seen.add(item['id'])
    if complete and seen != set(expected):
        raise ValueError('Capture omitted frozen training examples')
