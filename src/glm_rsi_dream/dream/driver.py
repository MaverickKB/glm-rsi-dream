"""One paired reload boundary, with an explicit offline filesystem driver."""

import copy
import fcntl
import subprocess
from pathlib import Path
from typing import Any

from ..files import read_json, sha, write_json


def validate_snapshot(snapshot: Any) -> Any:
    from .artifact import require_hash

    require_hash(snapshot["base_sha256"])
    ranks = snapshot["ranks"]
    if len(ranks) != 2:
        raise ValueError("A reload always owns both TP ranks")
    for rank in range(2):
        value = ranks[rank]
        if value["rank"] != rank or value["world"] != 2 or value["running"] is not True:
            raise ValueError("Expected a running ordered TP2 pair")
        if value["generation_api"] is not (rank == 0):
            raise ValueError("Only the head may own a generation API")
        for name in ("config_sha256", "image_sha256"):
            require_hash(value[name])
        if not value["container_id"]:
            raise ValueError("Exact rollback needs container identities")
    return snapshot


def target_for(snapshot: Any, manifest: Any, retire: Any = False) -> Any:
    target = copy.deepcopy(snapshot)
    if target["base_sha256"] != manifest["base_sha256"]:
        raise ValueError("Artifact belongs to a different frozen base")
    target["epoch"] = str(manifest["epoch"])
    for rank in range(2):
        target["ranks"][rank]["neurons_sha256"] = manifest["ranks"][rank]["sha256"]
        target["ranks"][rank]["active_memory_rows"] = 0
        target["ranks"][rank]["pocket_cache_clean"] = True
        for pocket in target["ranks"][rank]["pockets"].values():
            pocket["enabled"] = False
        if retire:
            for pocket in manifest["source_pockets"]:
                target["ranks"][rank]["pockets"].pop(pocket["id"], None)
    return target


class FilesystemPair:
    """CPU wiring only. It never connects to a model or claims native recall."""

    evidence_kind = "harness"

    def __init__(self, path: Any) -> None:
        self.path = Path(path)

    def snapshot(self) -> Any:
        return validate_snapshot(read_json(self.path))

    def prepare(self, manifest: Any, directory: Any, retire: Any = False) -> Any:
        return target_for(self.snapshot(), manifest, retire)

    def paired_reload(self, target: Any, *, expected: Any = None, recovery: Any = False) -> Any:
        validate_snapshot(target)
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not recovery and self.snapshot() != expected:
                raise ValueError("Paired runtime changed before reload")
            write_json(self.path, target)


class CommandPair:
    """Use the native owner's pinned paired guard runner, with JSON on stdin.

    The runner owns the real guard, independent recovery, container configuration,
    control-plane checks and app-level probes. This adapter owns no serving port.
    """

    evidence_kind = "native"

    def __init__(self, spec_path: Any) -> None:
        self.spec = read_json(spec_path)
        script = Path(self.spec["script"]).resolve()
        if sha(script) != self.spec["script_sha256"]:
            raise ValueError("Native paired runner content pin differs")
        self.script = script
        self.epoch = None

    def require_authority(self, epoch: Any) -> Any:
        from datetime import datetime

        from ..predream.timing import now_et

        queue = Path(self.spec["gpu_queue"]).read_text().splitlines()
        grant = self.spec["grant_line"]
        if grant not in queue or not grant.startswith("GRANT DREAM-trigger "):
            raise ValueError("Native work requires the owner's explicit DREAM-trigger grant")
        approval = read_json(self.spec["restart_approval"])
        if sha(self.spec["restart_approval"]) != self.spec["restart_approval_sha256"]:
            raise ValueError("Restart approval content pin differs")
        if approval.get("authority") != "owner approval" or approval.get("allow_paired_reload") is not True:
            raise ValueError("Native reload requires the owner approval receipt")
        expires = datetime.fromisoformat(approval["expires_et"])
        if expires.tzinfo is None or expires <= datetime.fromisoformat(now_et()):
            raise ValueError("Native restart approval expired")
        if epoch not in approval["epochs"]:
            raise ValueError("Restart approval does not cover this artifact epoch")

    def call(self, operation: Any, **values: Any) -> Any:
        import json

        if sha(self.script) != self.spec["script_sha256"]:
            raise ValueError("Native paired runner changed")
        request = {"schema_version": 1, "operation": operation, **values}
        result = subprocess.run([self.spec["python"], str(self.script)],
                                input=json.dumps(request), text=True, capture_output=True,
                                timeout=self.spec.get("timeout", 600), check=True)
        response = json.loads(result.stdout)
        if response.get("ok") is not True:
            raise RuntimeError("Native paired guard rejected: " + str(response))
        return response

    def snapshot(self) -> Any:
        return validate_snapshot(self.call("snapshot")["snapshot"])

    def prepare(self, manifest: Any, directory: Any, retire: Any = False) -> Any:
        self.require_authority(manifest["epoch"])
        self.epoch = manifest["epoch"]
        response = self.call("prepare", manifest=manifest, artifact_directory=str(directory), retire=retire)
        return validate_snapshot(response["target"])

    def paired_reload(self, target: Any, *, expected: Any = None, recovery: Any = False) -> Any:
        if not recovery:
            self.require_authority(self.epoch)
        self.call("guarded_paired_reload", target=validate_snapshot(target), expected=expected, recovery=recovery)
