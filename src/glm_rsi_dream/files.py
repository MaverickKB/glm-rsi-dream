"""Small durable-file and identity helpers."""
import hashlib
import json
import os
import tempfile
from pathlib import Path


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def invalid_constant(value):
    raise ValueError("nonfinite JSON number: " + value)


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=invalid_constant)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, name = tempfile.mkstemp(prefix="."+path.name+"-",suffix=".tmp",dir=path.parent)
    temporary = Path(name)
    with os.fdopen(descriptor, "w") as target:
        json.dump(value, target, indent=2, allow_nan=False)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)
