"""Load tunable component settings and preserve experiment identity."""

import hashlib
import json
from pathlib import Path
from typing import Any

from ..files import write_json


def load_policy(path: Any) -> Any:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("Policy config requires schema_version 1")
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Policy config requires a nonempty name")
    for section in ("extract", "grade"):
        if section not in value:
            value[section] = {}
        if not isinstance(value[section], dict):
            raise ValueError("Policy " + section + " section must be an object")
    return value


def record_policy(root: Any, policy: Any) -> Any:
    canonical = json.dumps(policy, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    source_hashes = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {"policy": policy, "policy_sha256": hashlib.sha256(canonical).hexdigest(),
              "component_source_sha256": source_hashes}
    path = Path(root) / "policy-config.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != record:
            raise ValueError("Configured experiment policy or source changed; use a new run root")
    else:
        write_json(path, record)
    return record
