"""Small durable-file and identity helpers."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def digest(value: Any) -> Any:
    # allow_nan would hash the tokens NaN and Infinity, which read_json rejects.
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def sha(path: Any) -> Any:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def invalid_constant(value: Any) -> Any:
    raise ValueError("nonfinite JSON number: " + value)


def read_json(path: Any) -> Any:
    return json.loads(Path(path).read_text(), parse_constant=invalid_constant)


def write_json(path: Any, value: Any) -> Any:
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
