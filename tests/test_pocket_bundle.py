"""Portable artifact round trips, consumer reads and recovery failures on CPU."""

import json
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from glm_rsi_dream.attention import attention
from glm_rsi_dream.export import FOUR_STREAMS, sha256
from glm_rsi_dream.files import digest, read_json, write_json
from glm_rsi_dream.pocket_bundle import export_bundle, import_bundle, verify_bundle
from tensorfold_dream.pocket_native import load


def make_source(root, hidden=8):
    root.mkdir(parents=True, exist_ok=True)
    base = root / "base"
    base.mkdir()
    config = {"model_type": "glm5_next", "text_config": {
        "model_type": "glm5_next_text", "hidden_size": hidden,
        "moe_intermediate_size": 2048, "n_shared_experts": 1, "n_routed_experts": 288,
        "num_hidden_layers": 45, "first_k_dense_replace": 3, "swiglu_limit": 10}}
    write_json(base / "config.json", config)
    write_json(base / "tokenizer.json", {"fixture": "fictional CPU test"})
    identity = root / "base-identity.json"
    write_json(identity, {"model": "GLM-5.3-Flash-EXL3", "version": "5.3-Flash",
                          "repository": "fictional/test-base", "revision": "a" * 40})
    torch.manual_seed(7301)
    pockets = []
    rank_roots = []
    additions = []
    for rank in (0, 1):
        directory = root / f"rank{rank}"
        directory.mkdir()
        rank_roots.append(directory)
        for key in ("Iris", "Rowan"):
            if rank == 0:
                pocket = {"key": key, "facts": key + " keeps a violet notebook.", "slots": 32,
                          "selected": [], "full": [], "router": {"owner": key,
                          "subjects": ["notebook"], "blocked": []}, "dev": []}
                pockets.append(pocket)
            else:
                pocket = pockets[0] if key == "Iris" else pockets[1]
            for kind, length in (("selected", 32), ("full", 96)):
                state = {"schema_version": 3, "mechanism": "attention_state_v1", "length": length,
                         "rows": [], "tail": []}
                for _ in range(11):
                    state["rows"].append(torch.randn(length, 4))
                for _ in range(11):
                    state["rows"].append(torch.randn(length // 4 + 1, 4))
                if kind == "selected":
                    state.update(rank=rank, training_only=True, dense_only=True)
                else:
                    state.update(rec=torch.randn(2, 4), conv=torch.randn(2, 4))
                path = directory / f"{key}-{kind}-rank{rank}.pt"
                torch.save(state, path)
                pocket[kind].append({"path": path.name, "sha256": sha256(path)})
        path = root / f"addition{rank}.pt"
        site = {"gate": torch.randn(2, hidden), "up": torch.randn(2, hidden), "down": torch.randn(hidden, 2)}
        torch.save({"schema_version": 1, "mechanism": "shared_ffn_swiglu_v1", "rank": rank,
                    "world": 2, "base_frozen": True, "layers": {15: site}}, path)
        additions.append(path)
    prefix = {"normal_ids": [1, 2], "teacher_ids": list(range(96))}
    for directory in rank_roots:
        write_json(directory / "prefix.json", prefix)
        binding = {"mechanism": "attention_state_v1", "tp_world": 2, "slots": 96,
                   "original_identities": ["b" * 64, "c" * 64],
                   "tokenizer_sha256": sha256(base / "tokenizer.json"),
                   "prefix_sha256": sha256(directory / "prefix.json")}
        pair = {"schema_version": 3, "binding": binding, "pair_sha256": digest(binding),
                "config_sha256": sha256(base / "config.json")}
        write_json(directory / "pair.json", pair)
        automatic = {"pair_sha256": pair["pair_sha256"], "status": "enabled",
                     "approved_by": "GLM", "pockets": pockets}
        automatic["revision"] = digest(automatic)[:16]
        write_json(directory / "automatic.json", automatic)
        write_json(directory / ("automatic-" + automatic["revision"] + ".json"), automatic)
    return rank_roots, base, identity, additions


def reseal(root):
    manifest = read_json(root / "manifest.json")
    for relative in manifest["files"]:
        path = root / relative
        manifest["files"][relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest.pop("bundle_sha256")
    manifest["bundle_sha256"] = digest(manifest)
    write_json(root / "manifest.json", manifest)


def four_stream_additions(paths, hidden):
    for path in paths:
        artifact = torch.load(path, weights_only=True)
        artifact["mechanism"] = FOUR_STREAMS
        artifact["study_sha256"] = digest({"purpose": "synthetic CPU four-stream transport"})
        for site in artifact["layers"].values():
            width = site["gate"].shape[0]
            site["down"] = torch.randn(4 * hidden, width)
        torch.save(artifact, path)


@pytest.mark.parametrize("hidden", [8, 4096])
def test_four_stream_bundle_relocates_all_outputs_exactly(tmp_path, hidden):
    roots, base, identity, additions = make_source(tmp_path / "source", hidden)
    four_stream_additions(additions, hidden)
    output = tmp_path / "bundle"
    exported = export_bundle(roots, base, identity, output, "four-stream-1", additions)
    moved = tmp_path / "moved"
    shutil.move(output, moved)
    boot = import_bundle(moved, tmp_path / "import", base, identity)
    assert boot["shared_mechanism"] == FOUR_STREAMS
    assert boot["shared_mode"] == "native_four_streams"
    installed = tmp_path / "import/bundle"
    binding = read_json(installed / "shared/manifest.json")["binding"]
    assert binding["output_contract"] == {
        "residual_streams": 4, "layout": "stream_major_hidden", "rank_reduction": "sum",
        "rank_order": [0, 1], "base_reduction": "separate_from_delta", "combine_orientation": "source_destination",
        "injection": "branch_before_hc_post", "branch_cast": "base_plus_delta_to_residual_dtype"}
    assert binding["study_sha256"] == torch.load(additions[0], weights_only=True)["study_sha256"]
    totals = []
    inputs = torch.randn(7, hidden) * 3
    for rank in (0, 1):
        before = torch.load(additions[rank], weights_only=True)["layers"][15]
        path = installed / f"shared/rank{rank}.pt"
        assert path.read_bytes() == additions[rank].read_bytes()
        assert (installed / f"attention/rank{rank}/rank{rank}.pt").read_bytes() == path.read_bytes()
        after = torch.load(path, weights_only=True)["layers"][15]
        for name in ("gate", "up", "down"):
            assert torch.equal(after[name], before[name])
        gate = torch.nn.functional.linear(inputs, after["gate"]).clamp(max=10)
        up = torch.nn.functional.linear(inputs, after["up"]).clamp(-10, 10)
        activation = torch.nn.functional.silu(gate) * up
        expected = torch.nn.functional.linear(activation, before["down"]).reshape(7, 4, hidden)
        actual = torch.nn.functional.linear(activation, after["down"]).reshape(7, 4, hidden)
        assert torch.equal(actual, expected)
        totals.append(actual)
    combined = totals[0] + totals[1]
    for stream in range(1, 4):
        assert not torch.equal(combined[:, 0], combined[:, stream])
    assert verify_bundle(installed, base, identity) == exported


@pytest.mark.parametrize("problem", ["shape", "mixed", "nonfinite", "site", "hc_mult",
                                    "study_missing", "study_malformed", "study_mismatch"])
def test_four_stream_invalid_contract_never_publishes(tmp_path, problem):
    roots, base, identity, additions = make_source(tmp_path / "source")
    four_stream_additions(additions, 8)
    value = torch.load(additions[1], weights_only=True)
    if problem == "shape":
        value["layers"][15]["down"] = torch.randn(8, 2)
    elif problem == "mixed":
        value["mechanism"] = "shared_ffn_swiglu_v1"
        value["layers"][15]["down"] = torch.randn(8, 2)
    elif problem == "nonfinite":
        value["layers"][15]["down"][24, 0] = float("nan")
    elif problem == "site":
        value["layers"][35] = value["layers"].pop(15)
    elif problem == "study_missing":
        value.pop("study_sha256")
    elif problem == "study_malformed":
        value["study_sha256"] = "unsealed"
    elif problem == "study_mismatch":
        value["study_sha256"] = digest({"purpose": "a different synthetic study"})
    else:
        config = read_json(base / "config.json")
        config["text_config"]["hc_mult"] = 2
        write_json(base / "config.json", config)
        for root in roots:
            pair = read_json(root / "pair.json")
            pair["config_sha256"] = sha256(base / "config.json")
            write_json(root / "pair.json", pair)
    torch.save(value, additions[1])
    with pytest.raises(ValueError):
        export_bundle(roots, base, identity, tmp_path / "bundle", "1", additions)
    assert not (tmp_path / "bundle").exists()


def test_four_stream_semantics_are_verified_even_after_reseal(tmp_path):
    roots, base, identity, additions = make_source(tmp_path / "source")
    four_stream_additions(additions, 8)
    output = tmp_path / "bundle"
    export_bundle(roots, base, identity, output, "1", additions)
    shared = read_json(output / "shared/manifest.json")
    shared["binding"]["output_contract"]["injection"] = "after_residual_mix"
    shared["pair_sha256"] = digest(shared["binding"])
    write_json(output / "shared/manifest.json", shared)
    reseal(output)
    with pytest.raises(ValueError, match="Shared addition manifest"):
        verify_bundle(output)


@pytest.mark.parametrize("include_shared", [False, True])
def test_move_import_reads_native_attention_and_preserves_numerical_outputs(tmp_path, include_shared):
    roots, base, identity, additions = make_source(tmp_path / "source")
    output = tmp_path / "bundle"
    shared = additions if include_shared else None
    exported = export_bundle(roots, base, identity, output, "1.0", shared)
    moved = tmp_path / "moved-server/bundle"
    moved.parent.mkdir()
    shutil.move(output, moved)
    target_base = tmp_path / "moved-server/base"
    shutil.copytree(base, target_base)
    target_identity = tmp_path / "moved-server/identity.json"
    shutil.copyfile(identity, target_identity)
    boot = import_bundle(moved, tmp_path / "moved-server/import-v1", target_base, target_identity)
    assert boot["bundle_sha256"] == exported["bundle_sha256"]
    assert boot["native_boot_verified"] is False
    assert bool(boot["shared_manifest"]) == include_shared
    installed = tmp_path / "moved-server/import-v1"
    for item in boot["ranks"]:
        rank = item["rank"]
        state = load(installed / item["attention_manifest"], SimpleNamespace(world=2))
        catalog = read_json(state["root"] / "automatic.json")
        for pocket in catalog["pockets"]:
            before = torch.load(roots[rank] / pocket["selected"][rank]["path"], weights_only=True)
            after = torch.load(state["root"] / pocket["selected"][rank]["path"], weights_only=True)
            queries = torch.randn(3, 2, 4)
            history = torch.randn(3, 4)
            for original, restored in zip(before["rows"], after["rows"]):
                assert torch.equal(original, restored)
                expected = attention(queries, original, history, torch.tensor([0, 1, 2]))
                actual = attention(queries, restored, history, torch.tensor([0, 1, 2]))
                assert torch.equal(actual, expected)
    assert verify_bundle(installed / "bundle", target_base, target_identity) == exported


def test_edit_reexport_and_recover_preserve_prior_import(tmp_path):
    roots, base, identity, _ = make_source(tmp_path / "source")
    first = tmp_path / "v1"
    export_bundle(roots, base, identity, first, "1")
    imported = tmp_path / "import-v1"
    old_boot = import_bundle(first, imported, base, identity)
    for root in roots:
        automatic = read_json(root / "automatic.json")
        automatic.pop("revision")
        automatic["pockets"][0]["facts"] = "Iris keeps a green notebook. Edit pending new learning."
        automatic["revision"] = digest(automatic)[:16]
        write_json(root / "automatic.json", automatic)
        write_json(root / ("automatic-" + automatic["revision"] + ".json"), automatic)
    second = tmp_path / "v2"
    updated = export_bundle(roots, base, identity, second, "2")
    assert updated["bundle_sha256"] != old_boot["bundle_sha256"]
    import_bundle(second, tmp_path / "import-v2", base, identity)
    recovered = import_bundle(first, tmp_path / "recovered-v1", base, identity)
    assert recovered == old_boot
    assert read_json(imported / "boot.json") == old_boot
    with pytest.raises(FileExistsError):
        import_bundle(second, imported, base, identity)


@pytest.mark.parametrize("problem", ["config", "tokenizer", "revision"])
def test_wrong_target_base_rejected_before_import(tmp_path, problem):
    roots, base, identity, _ = make_source(tmp_path / "source")
    output = tmp_path / "bundle"
    export_bundle(roots, base, identity, output, "1")
    if problem == "revision":
        value = read_json(identity)
        value["revision"] = "d" * 40
        write_json(identity, value)
    else:
        with (base / (problem + ".json")).open("a") as target:
            target.write(" ")
    with pytest.raises(ValueError, match="Target base"):
        import_bundle(output, tmp_path / "imported", base, identity)
    assert not (tmp_path / "imported").exists()


@pytest.mark.parametrize("problem", ["corrupt", "extra", "nested_manifest", "symlink", "manifest_version", "path"])
def test_bundle_integrity_and_path_boundaries(tmp_path, problem):
    roots, base, identity, _ = make_source(tmp_path / "source")
    output = tmp_path / "bundle"
    export_bundle(roots, base, identity, output, "1")
    path = output / "attention/rank0/Iris-selected-rank0.pt"
    if problem == "corrupt":
        path.write_bytes(b"damaged")
    elif problem in ("extra", "nested_manifest"):
        name = "manifest.json" if problem == "nested_manifest" else "unlisted.txt"
        (output / "attention" / name).write_text("unexpected")
    elif problem == "symlink":
        path.unlink()
        path.symlink_to(roots[0] / path.name)
    else:
        manifest = read_json(output / "manifest.json")
        if problem == "manifest_version":
            manifest["version"] = "unexpected"
        else:
            manifest["files"]["../outside.pt"] = manifest["files"].pop("attention/rank0/Iris-selected-rank0.pt")
            manifest.pop("bundle_sha256")
            manifest["bundle_sha256"] = digest(manifest)
        write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError):
        verify_bundle(output)


@pytest.mark.parametrize("problem", ["rank", "nonfinite", "full_missing", "catalog", "width", "one_rank", "pooled_row"])
def test_malformed_source_never_publishes(tmp_path, problem):
    roots, base, identity, additions = make_source(tmp_path / "source")
    shared = additions
    if problem in ("rank", "nonfinite", "pooled_row"):
        path = roots[1] / "Iris-selected-rank1.pt"
        value = torch.load(path, weights_only=True)
        if problem == "rank":
            value["rank"] = 0
        elif problem == "nonfinite":
            value["rows"][0][0, 0] = float("nan")
        else:
            value["rows"][11] = torch.randn(32, 4)
        torch.save(value, path)
        for root in roots:
            automatic = read_json(root / "automatic.json")
            automatic.pop("revision")
            automatic["pockets"][0]["selected"][1]["sha256"] = sha256(path)
            automatic["revision"] = digest(automatic)[:16]
            write_json(root / "automatic.json", automatic)
            write_json(root / ("automatic-" + automatic["revision"] + ".json"), automatic)
    elif problem == "full_missing":
        (roots[1] / "Iris-full-rank1.pt").unlink()
    elif problem == "catalog":
        value = read_json(roots[1] / "automatic.json")
        value.pop("revision")
        value["status"] = "disabled"
        value["revision"] = digest(value)[:16]
        write_json(roots[1] / "automatic.json", value)
        write_json(roots[1] / ("automatic-" + value["revision"] + ".json"), value)
    elif problem == "width":
        value = torch.load(additions[1], weights_only=True)
        value["layers"][15] = {"gate": torch.randn(3, 8), "up": torch.randn(3, 8), "down": torch.randn(8, 3)}
        torch.save(value, additions[1])
    else:
        shared = [additions[0]]
    with pytest.raises((ValueError, OSError)):
        export_bundle(roots, base, identity, tmp_path / "bundle", "1", shared)
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle-*"))


def test_corrupt_semantic_payload_rejected_even_after_outer_reseal(tmp_path):
    roots, base, identity, additions = make_source(tmp_path / "source")
    output = tmp_path / "bundle"
    export_bundle(roots, base, identity, output, "1", additions)
    value = read_json(output / "shared/manifest.json")
    value["binding"]["world"] = 1
    write_json(output / "shared/manifest.json", value)
    reseal(output)
    with pytest.raises(ValueError, match="Shared addition manifest"):
        verify_bundle(output)


def test_copy_failure_cleans_staging_and_preserves_source(tmp_path, monkeypatch):
    roots, base, identity, _ = make_source(tmp_path / "source")
    original = (roots[0] / "pair.json").read_bytes()
    real_copy = shutil.copyfile
    calls = 0

    def fail_copy(source, destination):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("simulated interrupted copy")
        return real_copy(source, destination)

    monkeypatch.setattr(shutil, "copyfile", fail_copy)
    with pytest.raises(OSError, match="interrupted"):
        export_bundle(roots, base, identity, tmp_path / "bundle", "1")
    assert (roots[0] / "pair.json").read_bytes() == original
    assert not list(tmp_path.glob(".bundle-*"))
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("four_streams", [False, True])
def test_cli_export_verify_import_and_rejection(tmp_path, four_streams):
    roots, base, identity, additions = make_source(tmp_path / "source")
    if four_streams:
        four_stream_additions(additions, 8)
    output = tmp_path / "bundle"
    command = [sys.executable, "-m", "glm_rsi_dream.pocket_bundle"]
    options = ["--base", str(base), "--base-identity", str(identity)]
    result = subprocess.run(command + ["export", "--attention-rank0", str(roots[0]),
                            "--attention-rank1", str(roots[1]), "--rank0", str(additions[0]),
                            "--rank1", str(additions[1]), "--version", "1", "--output", str(output)] + options,
                            capture_output=True, text=True, check=True)
    manifest = json.loads(result.stdout)
    checked = subprocess.run(command + ["verify", str(output)] + options,
                             capture_output=True, text=True, check=True)
    assert json.loads(checked.stdout) == manifest
    imported = subprocess.run(command + ["import", str(output), "--output", str(tmp_path / "import")] + options,
                              capture_output=True, text=True, check=True)
    assert json.loads(imported.stdout)["bundle_sha256"] == manifest["bundle_sha256"]
    failed = subprocess.run(command + ["import", str(output), "--output", str(tmp_path / "import")] + options,
                            capture_output=True, text=True)
    assert failed.returncode == 1
    assert "fresh import" in failed.stderr
