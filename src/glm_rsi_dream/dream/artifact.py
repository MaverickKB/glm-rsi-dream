"""Content seals and strict GLM shared-expert artifact validation."""

import copy
import re
from pathlib import Path
from typing import Any

from ..files import digest, read_json, sha, write_json
from ..pocket_off_eval import digest as evaluation_digest
from ..pocket_off_eval import (
    reply_signature,
    request_for,
    semantic_pass,
    validate_runtime,
    validate_suite,
)

GEOMETRY = {"hidden": 4096, "intermediate": 2048, "shared_experts": 1, "routed_experts": 288}


def require_hash(value: Any) -> Any:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected a SHA256 identity")
    return value


def safe_id(value: Any) -> Any:
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}", value):
        raise ValueError("Expected a safe single-component id")
    return value


def seal(value: Any) -> Any:
    result = copy.deepcopy(value)
    result.pop("manifest_sha256", None)
    result["manifest_sha256"] = digest(result)
    return result


def verify_seal(value: Any) -> Any:
    content = copy.deepcopy(value)
    expected = content.pop("manifest_sha256", None)
    if digest(content) != expected:
        raise ValueError("Manifest content seal differs")


def checked_file(root: Any, entry: Any) -> Any:
    path = (Path(root) / entry["path"]).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError("Missing or escaped artifact file")
    if sha(path) != require_hash(entry["sha256"]):
        raise ValueError("Artifact file hash differs")
    return path


def validate_tensors(path: Any, rank: Any) -> Any:
    import torch

    packet = torch.load(path, weights_only=True, map_location="cpu")
    if packet.get("schema_version") != 1 or packet.get("mechanism") != "shared_ffn_swiglu_v1":
        raise ValueError("Expected shared-expert SwiGLU addition")
    if packet.get("rank") != rank or packet.get("world") != 2 or packet.get("base_frozen") is not True:
        raise ValueError("Frozen-base paired rank identity differs")
    if not packet.get("layers"):
        raise ValueError("Shared-expert addition contains no layers")
    shapes = {}
    for layer, site in packet["layers"].items():
        if type(layer) is not int or not 3 <= layer <= 39 or set(site) != {"gate", "up", "down"}:
            raise ValueError("Expected middle shared-expert gate/up/down tensors")
        gate, up, down = site["gate"], site["up"], site["down"]
        if gate.ndim != 2 or gate.shape[1] != 4096 or gate.shape[0] <= 0:
            raise ValueError("GLM hidden dimension must be 4096")
        if up.shape != gate.shape or down.shape != (4096, gate.shape[0]):
            raise ValueError("Shared-expert tensor shapes differ")
        for tensor in (gate, up, down):
            if not torch.isfinite(tensor).all():
                raise ValueError("Nonfinite shared-expert tensor")
        shapes[str(layer)] = gate.shape[0]
    return shapes


def load_artifact(directory: Any, qualified: Any = True) -> Any:
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    verify_seal(manifest)
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "shared_expert_addition":
        raise ValueError("Unsupported dream artifact version")
    if manifest.get("model") != "GLM-5.3-Flash-EXL3" or manifest.get("geometry") != GEOMETRY:
        raise ValueError("Only verified GLM-5.3-Flash geometry is supported")
    if type(manifest.get("epoch")) is not int or manifest["epoch"] < 1:
        raise ValueError("Artifact needs a positive epoch")
    require_hash(manifest["base_sha256"])
    if sha(directory / "training.json") != manifest["training_sha256"]:
        raise ValueError("Training plan content seal differs")
    if not manifest["source_pockets"] or len(manifest["ranks"]) != 2:
        raise ValueError("Artifact requires source pockets and both ranks")
    shapes = None
    for rank in range(2):
        entry = manifest["ranks"][rank]
        if entry["rank"] != rank:
            raise ValueError("Artifact ranks must be ordered 0, 1")
        current = validate_tensors(checked_file(directory, entry), rank)
        if shapes is not None and current != shapes:
            raise ValueError("Paired layer placement or width differs")
        shapes = current
    if shapes != manifest["width_per_rank"]:
        raise ValueError("Artifact width metadata differs from tensors")
    if qualified:
        checked_file(directory, manifest["evaluation"])
        checked_file(directory, manifest["baseline"])
        checked_file(directory, manifest["suite"])
        validate_evaluation(directory, manifest)
    return manifest


def validate_evaluation(directory: Any, manifest: Any) -> Any:
    """Recheck raw evidence, never trust a summary passed flag alone."""
    report = read_json(checked_file(directory, manifest["evaluation"]))
    baseline = read_json(checked_file(directory, manifest["baseline"]))
    suite = read_json(checked_file(directory, manifest["suite"]))
    validate_suite(suite)
    baseline_replies = {}
    groups = set()
    for phase, value in (("baseline", baseline), ("consolidated", report)):
        content = copy.deepcopy(value)
        expected = content.pop("report_sha256", None)
        if evaluation_digest(content) != expected:
            raise ValueError("Evaluation content seal differs")
        if value.get("phase") != phase or value.get("complete") is not True or value.get("runtime_verified") is not True:
            raise ValueError("Pocket-OFF evaluation requires complete runtime evidence")
        if value["suite_sha256"] != evaluation_digest(suite) or len(value["records"]) != len(suite["cases"]):
            raise ValueError("Evaluation must cover the full sealed suite")
        stable = None
        for case, record in zip(suite["cases"], value["records"]):
            groups.add(case["group"])
            if record["id"] != case["id"] or record.get("error"):
                raise ValueError("Missing, failed or reordered evaluation observation")
            expected_request = request_for(case, suite.get("parameters", {}), record["probe_id"])
            actual_reply = reply_signature(record["reply"])
            if record["request"] != expected_request or actual_reply != record["reply_signature"]:
                raise ValueError("Raw evaluation request or reply differs")
            for snapshot in (record["runtime_before"], record["runtime_after"]):
                current = validate_runtime(snapshot, record["probe_id"], phase)
                if stable is not None and current != stable:
                    raise ValueError("Runtime changed during the evaluation phase")
                stable = current
                if snapshot["base_sha256"] != manifest["base_sha256"]:
                    raise ValueError("Evaluation base differs from artifact")
                if phase == "consolidated":
                    if snapshot["epoch"] != str(manifest["epoch"]):
                        raise ValueError("Evaluation epoch differs from artifact")
                    for rank in snapshot["ranks"]:
                        if rank["neurons_sha256"] != manifest["ranks"][rank["tp_rank"]]["sha256"]:
                            raise ValueError("Evaluation measured a different rank artifact")
            if phase == "baseline":
                baseline_replies[case["id"]] = actual_reply
            else:
                if case["kind"] == "exact":
                    passed = actual_reply == baseline_replies[case["id"]]
                else:
                    passed = semantic_pass(case, record["judgment"])
                if not passed or record.get("passed") is not True:
                    raise ValueError("Raw recall, control or leakage evaluation failed")
    if report.get("passed") is not True or report.get("baseline_comparison") is not True:
        raise ValueError("Pocket-OFF evaluation must pass with a baseline comparison")
    required = {"ordinary", "protected", "leak"}
    for pocket in manifest["source_pockets"]:
        verify_seal(pocket)
        required.add(pocket["recall_group"])
        count = 0
        statements = []
        for case in suite["cases"]:
            if case["group"] == pocket["recall_group"]:
                if case["kind"] != "recall":
                    raise ValueError("Pocket recall group must contain recall questions")
                count += 1
                statements.append(str(case["facts"]))
        if count < pocket["minimum_recall"]:
            raise ValueError("Recall coverage is below the accepted pocket bar")
        for item in pocket["items"]:
            covered = False
            for facts in statements:
                if item["statement"] in facts:
                    covered = True
            if not covered:
                raise ValueError("Recall rubric lacks an accepted source fact")
    if not required <= groups:
        raise ValueError("Evaluation lacks recall, ordinary, protected or leak groups")
    for case in suite["cases"]:
        if case["group"] in ("ordinary", "protected") and case["kind"] != "exact":
            raise ValueError("Ordinary and protected controls require exact comparison")
        if case["group"] == "leak" and case["kind"] != "leak":
            raise ValueError("Leak group requires semantic leakage checks")
        if case["group"] == "precision" and case["kind"] != "recall":
            raise ValueError("Precision questions require factual recall")
    if manifest["evidence_kind"] == "native":
        counts = {"ordinary": 0, "protected": 0, "leak": 0, "precision": 0}
        for case in suite["cases"]:
            if case["group"] in counts:
                counts[case["group"]] += 1
        for group, minimum in (("ordinary", 60), ("protected", 8), ("leak", 12), ("precision", 24)):
            if counts[group] < minimum:
                raise ValueError("Native evaluation lacks the full " + group + " acceptance bar")
    return report


def publish_manifest(directory: Any, value: Any) -> Any:
    manifest = seal(value)
    write_json(Path(directory) / "manifest.json", manifest)
    return manifest
