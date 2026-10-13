"""Portable, versioned attention pockets and optional paired shared additions."""

import argparse
import io
import json
import pickle
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import torch

from .export import FOUR_STREAMS, dimensions, read_additions, sha256
from .files import digest, read_json, write_json
from .manifest import MANIFEST

FORMAT = "glm_personalization_bundle_v1"


def safe_file(root: Any, relative: Any) -> Any:
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("Artifact path must be a portable relative path")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or not parts.parts or ".." in parts.parts:
        raise ValueError("Artifact path escapes its root")
    root = Path(root).resolve()
    path = root
    for part in parts.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Artifact symlinks are forbidden")
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError("Artifact path is missing or outside its root")
    return path


def base_binding(base: Any, identity_path: Any) -> Any:
    identity = read_json(identity_path)
    if not isinstance(identity, dict) or set(identity) != {"model", "version", "repository", "revision"}:
        raise ValueError("Base identity requires model, version, repository and revision")
    for name in ("model", "version", "repository"):
        if not isinstance(identity[name], str) or not identity[name].strip():
            raise ValueError("Base identity fields must be nonempty text")
    if identity["model"] != MANIFEST["model"]:
        raise ValueError("Only the observed GLM-5.3-Flash model is supported")
    if not isinstance(identity["revision"], str) or not re.fullmatch(r"(?:[a-f0-9]{40}|[a-f0-9]{64})", identity["revision"]):
        raise ValueError("Base revision must be an immutable hexadecimal identity")
    result = dict(identity)
    for name in ("config.json", "tokenizer.json"):
        result[name.removesuffix(".json") + "_sha256"] = sha256(safe_file(base, name))
    return result


def check_tensor_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise ValueError("Attention tensors must be dense")
        if value.is_floating_point() and not torch.isfinite(value.float()).all():
            raise ValueError("Attention tensor contains a nonfinite value")
    elif isinstance(value, dict):
        for item in value.values():
            check_tensor_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            check_tensor_tree(item)


def attention_tensor(path: Any, rank: Any, kind: Any, slots: Any) -> Any:
    raw = path.read_bytes()
    value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if (not isinstance(value, dict) or value.get("schema_version") != 3
            or value.get("mechanism") != "attention_state_v1"):
        raise ValueError("Attention state schema mismatch")
    length = value.get("length")
    if type(length) is not int or length < slots:
        raise ValueError("Attention state length mismatch")
    if kind == "selected":
        if (type(value.get("rank")) is not int or value["rank"] != rank
                or length != slots or value.get("training_only") is not True
                or value.get("dense_only") is not True):
            raise ValueError("Selected attention rank or geometry mismatch")
    else:
        for name in ("rec", "conv"):
            if not isinstance(value.get(name), torch.Tensor):
                raise ValueError("Full attention state needs recurrence and convolution")
    rows = value.get("rows")
    dense_sites = len(MANIFEST["memory"]["layers"])
    if not isinstance(rows, list) or len(rows) < dense_sites or not isinstance(value.get("tail"), list):
        raise ValueError("Attention state needs rows and tail")
    for index, row in enumerate(rows):
        expected_rows = length if index < dense_sites else length // 4 + 1
        if not isinstance(row, torch.Tensor) or row.ndim < 2 or row.shape[0] != expected_rows:
            raise ValueError("Attention row geometry mismatch")
    check_tensor_tree(value)
    return value


def attention_files(root: Any, rank: Any, base: Any) -> Any:
    pair = read_json(safe_file(root, "pair.json"))
    binding = pair["binding"]
    if (pair.get("schema_version") != 3 or binding.get("mechanism") != "attention_state_v1"
            or binding.get("tp_world") != 2 or pair.get("pair_sha256") != digest(binding)
            or pair.get("config_sha256") != base["config_sha256"]
            or binding.get("tokenizer_sha256") != base["tokenizer_sha256"]
            or binding.get("prefix_sha256") != sha256(safe_file(root, "prefix.json"))):
        raise ValueError("Attention pair or base identity mismatch")
    prefix = read_json(root / "prefix.json")
    for name in ("teacher_ids", "normal_ids"):
        ids = prefix.get(name)
        if not isinstance(ids, list) or not ids:
            raise ValueError("Attention prefix needs token IDs")
        for token in ids:
            if type(token) is not int or token < 0:
                raise ValueError("Attention prefix token ID mismatch")
    automatic = read_json(safe_file(root, "automatic.json"))
    if automatic.get("status") not in ("enabled", "disabled") or automatic.get("approved_by") != "GLM":
        raise ValueError("Attention catalog status or approval is missing")
    sealed = dict(automatic)
    revision = sealed.pop("revision", None)
    if (automatic.get("pair_sha256") != pair["pair_sha256"]
            or not isinstance(revision, str) or digest(sealed)[:16] != revision):
        raise ValueError("Automatic pocket revision mismatch")
    version_path = "automatic-" + revision + ".json"
    if read_json(safe_file(root, version_path)) != automatic:
        raise ValueError("Immutable automatic pocket revision differs")
    pockets = automatic.get("pockets")
    if not isinstance(pockets, list) or not pockets:
        raise ValueError("At least one attention pocket is required")
    names = ["pair.json", "prefix.json", "automatic.json", version_path]
    keys = set()
    for pocket in pockets:
        key = pocket.get("key")
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9]{1,24}", key) or key in keys:
            raise ValueError("Attention pocket key is invalid or repeated")
        keys.add(key)
        if not isinstance(pocket.get("facts"), str) or not pocket["facts"].strip():
            raise ValueError("Editable pocket facts must be nonempty text")
        routers = pocket.get("routers") or [pocket.get("router")]
        for router in routers:
            if (not isinstance(router, dict) or not isinstance(router.get("owner"), str)
                    or not router["owner"].strip()):
                raise ValueError("Pocket admission needs an owner")
            for name in ("subjects", "blocked", "owners", "entities", "referents", "domains", "relations"):
                if name not in router and name not in ("subjects", "blocked"):
                    continue
                phrases = router.get(name)
                if not isinstance(phrases, list) or (name == "subjects" and not phrases):
                    raise ValueError("Pocket admission rule is missing " + name)
                for phrase in phrases:
                    if not isinstance(phrase, str) or not phrase.strip():
                        raise ValueError("Pocket admission rule contains an empty phrase")
        slots = pocket.get("slots")
        if type(slots) is not int or slots <= 0 or slots % 32 or slots > len(prefix["teacher_ids"]):
            raise ValueError("Attention pocket slots differ from native alignment")
        states = {}
        for kind in ("selected", "full"):
            entries = pocket.get(kind)
            if not isinstance(entries, list) or len(entries) != 2:
                raise ValueError("Every pocket needs selected and full files for both ranks")
            entry = entries[rank]
            path = safe_file(root, entry["path"])
            if sha256(path) != entry.get("sha256"):
                raise ValueError("Attention artifact hash mismatch")
            states[kind] = attention_tensor(path, rank, kind, slots)
            names.append(entry["path"])
        if len(states["selected"]["rows"]) != len(states["full"]["rows"]):
            raise ValueError("Selected and full attention site counts differ")
        for selected, full in zip(states["selected"]["rows"], states["full"]["rows"]):
            if selected.shape[1:] != full.shape[1:] or selected.dtype != full.dtype:
                raise ValueError("Selected and full attention row formats differ")
    return names, pair, automatic


def shared_binding(paths: Any, config_path: Any) -> Any:
    config = read_json(config_path)
    text, _, limit = dimensions(config)
    artifacts = read_additions(paths, text, limit)
    layers = [artifacts[0]["layers"], artifacts[1]["layers"]]
    geometry = {}
    for name in ("hidden_size", "moe_intermediate_size", "n_shared_experts",
                 "n_routed_experts", "num_hidden_layers", "first_k_dense_replace"):
        value = text.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError("Shared addition base geometry is missing " + name)
        geometry[name] = value
    for index in layers[0]:
        if not geometry["first_k_dense_replace"] <= index <= min(39, geometry["num_hidden_layers"] - 1):
            raise ValueError("Shared addition must target middle MoE sites")
        if layers[0][index]["gate"].shape != layers[1][index]["gate"].shape:
            raise ValueError("Shared addition rank widths differ")
        for rank in (0, 1):
            for tensor in layers[rank][index].values():
                if tensor.dtype != torch.float32:
                    raise ValueError("Inference-only loader requires FP32 additions")
    if limit != 10:
        raise ValueError("Inference-only loader requires SwiGLU limit 10")
    ranks = []
    for rank, path in enumerate(paths):
        ranks.append({"rank": rank, "path": f"rank{rank}.pt", "sha256": sha256(path)})
    binding = {"mechanism": artifacts[0]["mechanism"], "model": MANIFEST["model"],
               "world": 2, "model_config_sha256": sha256(config_path),
               "geometry": geometry, "ranks": ranks}
    if artifacts[0]["mechanism"] == FOUR_STREAMS:
        binding["study_sha256"] = artifacts[0]["study_sha256"]
        binding["output_contract"] = {
            "residual_streams": 4, "layout": "stream_major_hidden",
            "rank_reduction": "sum", "rank_order": [0, 1], "base_reduction": "separate_from_delta",
            "combine_orientation": "source_destination", "injection": "branch_before_hc_post",
            "branch_cast": "base_plus_delta_to_residual_dtype",
        }
    return {"schema_version": 1, "binding": binding, "pair_sha256": digest(binding)}


def verify_bundle(root: Any, base: Any = None, identity: Any = None) -> Any:
    root = Path(root).resolve()
    manifest = read_json(safe_file(root, "manifest.json"))
    sealed = dict(manifest)
    expected = sealed.pop("bundle_sha256", None)
    if (manifest.get("schema_version") != 1 or manifest.get("format") != FORMAT
            or expected != digest(sealed) or not isinstance(manifest.get("version"), str)
            or not manifest["version"].strip() or manifest.get("world") != 2):
        raise ValueError("Bundle manifest identity or version mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Bundle needs an explicit file inventory")
    for relative, entry in files.items():
        path = safe_file(root, relative)
        if path.stat().st_size != entry.get("bytes") or sha256(path) != entry.get("sha256"):
            raise ValueError("Bundle artifact hash or size mismatch: " + relative)
    actual_files = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Bundle symlinks are forbidden")
        relative = path.relative_to(root).as_posix()
        if path.is_file() and relative != "manifest.json":
            actual_files.add(relative)
    if actual_files != set(files):
        raise ValueError("Bundle contains unlisted or missing files")
    if (base is None) != (identity is None):
        raise ValueError("Supply both target base and target identity")
    if base is not None and base_binding(base, identity) != manifest["base"]:
        raise ValueError("Target base identity differs from bundle")
    embedded_base = base_binding(root / "base", root / "base/identity.json")
    if embedded_base != manifest["base"]:
        raise ValueError("Embedded base identity differs from manifest")
    previous = None
    required = {"base/config.json", "base/tokenizer.json", "base/identity.json", "runtime-contract.json"}
    for rank in (0, 1):
        names, pair, automatic = attention_files(root / f"attention/rank{rank}", rank, manifest["base"])
        current = (pair, automatic)
        if previous is not None and current != previous:
            raise ValueError("Attention ranks disagree on pair or pocket catalog")
        previous = current
        for name in names:
            required.add(f"attention/rank{rank}/" + name)
    if manifest.get("shared_addition") is True:
        shared = root / "shared"
        paired = shared_binding([safe_file(shared, "rank0.pt"), safe_file(shared, "rank1.pt")], root / "base/config.json")
        if read_json(safe_file(shared, "manifest.json")) != paired:
            raise ValueError("Shared addition manifest mismatch")
        required.update({"shared/rank0.pt", "shared/rank1.pt", "shared/manifest.json"})
        for rank in (0, 1):
            attention = root / f"attention/rank{rank}"
            activation = {"by_rank": paired["binding"]["ranks"]}
            if read_json(safe_file(attention, "neurons-current.json")) != activation:
                raise ValueError("Attention shared addition activation differs")
            if sha256(safe_file(attention, f"rank{rank}.pt")) != paired["binding"]["ranks"][rank]["sha256"]:
                raise ValueError("Attention shared addition payload differs")
            required.update({f"attention/rank{rank}/neurons-current.json", f"attention/rank{rank}/rank{rank}.pt"})
    elif manifest.get("shared_addition") is not False:
        raise ValueError("Shared addition presence must be explicit")
    if required != set(files):
        raise ValueError("Bundle inventory differs from runtime artifacts")
    return manifest


def export_bundle(attention_roots: Any, base: Any, identity: Any, output: Any, version: Any, additions: Any = None) -> Any:
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a fresh bundle directory")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("Bundle version is required")
    if len(attention_roots) != 2:
        raise ValueError("Both attention rank directories are required")
    binding = base_binding(base, identity)
    catalogs = []
    for rank, root in enumerate(attention_roots):
        catalogs.append(attention_files(Path(root), rank, binding))
    if catalogs[0][1:] != catalogs[1][1:]:
        raise ValueError("Attention ranks disagree on pair or pocket catalog")
    shared = None
    if additions is not None:
        shared = shared_binding(additions, Path(base) / "config.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + output.name + "-", dir=output.parent))
    try:
        for rank, root in enumerate(attention_roots):
            for relative in catalogs[rank][0]:
                target = staging / f"attention/rank{rank}" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(safe_file(root, relative), target)
        (staging / "base").mkdir()
        for name in ("config.json", "tokenizer.json"):
            shutil.copyfile(safe_file(base, name), staging / "base" / name)
        shutil.copyfile(identity, staging / "base/identity.json")
        write_json(staging / "runtime-contract.json", MANIFEST)
        if shared is not None:
            (staging / "shared").mkdir()
            for rank, path in enumerate(additions):
                shutil.copyfile(path, staging / "shared" / f"rank{rank}.pt")
                attention = staging / f"attention/rank{rank}"
                target = attention / f"rank{rank}.pt"
                if target.exists():
                    raise ValueError("Attention artifact collides with shared addition filename")
                shutil.copyfile(path, target)
                write_json(attention / "neurons-current.json", {"by_rank": shared["binding"]["ranks"]})
            write_json(staging / "shared/manifest.json", shared)
        files = {}
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                files[path.relative_to(staging).as_posix()] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        manifest = {"schema_version": 1, "format": FORMAT, "version": version,
                    "base": binding, "world": 2, "shared_addition": shared is not None,
                    "files": files, "producer_version": MANIFEST["version"]}
        manifest["bundle_sha256"] = digest(manifest)
        write_json(staging / "manifest.json", manifest)
        verify_bundle(staging)
        if output.exists():
            raise FileExistsError("Output appeared while exporting")
        staging.rename(output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def import_bundle(source: Any, output: Any, base: Any, identity: Any) -> Any:
    manifest = verify_bundle(source, base, identity)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a fresh import directory; earlier versions remain recoverable")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + output.name + "-", dir=output.parent))
    try:
        payload = staging / "bundle"
        payload.mkdir()
        for relative in ["manifest.json"] + list(manifest["files"]):
            target = payload / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_file(source, relative), target)
        verify_bundle(payload, base, identity)
        ranks = []
        for rank in (0, 1):
            ranks.append({"rank": rank, "attention_manifest": f"bundle/attention/rank{rank}/pair.json",
                          "attention_root": f"bundle/attention/rank{rank}"})
        boot = {"schema_version": 1, "bundle_sha256": manifest["bundle_sha256"],
                "version": manifest["version"], "base": manifest["base"], "world": 2,
                "ranks": ranks, "shared_manifest": "bundle/shared/manifest.json" if manifest["shared_addition"] else None,
                "attention_environment": "TF_DREAM_EVAL_MANIFEST", "shared_environment": "TF_GLM_DREAM_MANIFEST",
                "shared_mode": "attention_neurons_current" if manifest["shared_addition"] else None,
                "generation_rank": 0, "native_boot_verified": False}
        if manifest["shared_addition"]:
            shared = read_json(payload / "shared/manifest.json")
            boot["shared_mechanism"] = shared["binding"]["mechanism"]
            if shared["binding"]["mechanism"] == FOUR_STREAMS:
                boot["shared_mode"] = "native_four_streams"
        write_json(staging / "boot.json", boot)
        if output.exists():
            raise FileExistsError("Output appeared while importing")
        staging.rename(output)
        return boot
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--attention-rank0", type=Path, required=True)
    export.add_argument("--attention-rank1", type=Path, required=True)
    export.add_argument("--rank0", type=Path)
    export.add_argument("--rank1", type=Path)
    export.add_argument("--version", required=True)
    export.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("bundle", type=Path)
    restore = commands.add_parser("import")
    restore.add_argument("bundle", type=Path)
    restore.add_argument("--output", type=Path, required=True)
    for command in (export, verify, restore):
        required = command is not verify
        command.add_argument("--base", type=Path, required=required)
        command.add_argument("--base-identity", type=Path, required=required)
    args = parser.parse_args()
    try:
        if args.command == "export":
            if (args.rank0 is None) != (args.rank1 is None):
                raise ValueError("Supply both shared addition ranks")
            additions = None
            if args.rank0 is not None:
                additions = [args.rank0, args.rank1]
            result = export_bundle([args.attention_rank0, args.attention_rank1], args.base,
                                   args.base_identity, args.output, args.version, additions)
        elif args.command == "verify":
            result = verify_bundle(args.bundle, args.base, args.base_identity)
        else:
            result = import_bundle(args.bundle, args.output, args.base, args.base_identity)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        parser.exit(1, "Bundle rejected: " + str(error) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
