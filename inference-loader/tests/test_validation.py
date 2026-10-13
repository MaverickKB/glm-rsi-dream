import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from dream_loader import runtime
from dream_loader.artifact import geometry, identity, load_pair
from dream_loader.bundle import bundle


def resign(path, manifest):
    manifest["pair_sha256"] = identity(manifest["binding"])
    path.write_text(json.dumps(manifest))


def test_bundle_preserves_exact_engine_artifact_bytes(addition):
    path, _, manifest = addition
    for rank in range(2):
        source = path.parent.parent / f"original-rank{rank}.pt"
        assert source.read_bytes() == (path.parent / f"rank{rank}.pt").read_bytes()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["binding"]["ranks"][rank]["sha256"]


@pytest.mark.parametrize("change", ["rank", "world", "mechanism", "base", "nan", "shape",
                                    "dtype", "layer", "site", "empty", "width"])
def test_reject_invalid_artifacts(addition, change):
    path, _, manifest = addition
    rank_file = path.parent / "rank0.pt"
    value = torch.load(rank_file, weights_only=True)
    site = value["layers"][11]
    if change == "rank":
        value["rank"] = 1
    elif change == "world":
        value["world"] = 1
    elif change == "mechanism":
        value["mechanism"] = "routed_expert"
    elif change == "base":
        value["base_frozen"] = False
    elif change == "nan":
        site["gate"][0, 0] = float("nan")
    elif change == "shape":
        site["down"] = site["down"].T.contiguous()
    elif change == "dtype":
        site["gate"] = site["gate"].bfloat16()
    elif change == "layer":
        value["layers"][2] = value["layers"].pop(11)
    elif change == "site":
        site["router"] = torch.zeros(1, 1)
    elif change == "empty":
        value["layers"] = {}
    elif change == "width":
        value["layers"][11] = {"gate": site["gate"][:16], "up": site["up"][:16],
                               "down": site["down"][:, :16]}
    torch.save(value, rank_file)
    manifest["binding"]["ranks"][0]["sha256"] = hashlib.sha256(rank_file.read_bytes()).hexdigest()
    resign(path, manifest)
    with pytest.raises(ValueError):
        load_pair(path)


@pytest.mark.parametrize("change", ["bytes", "identity", "path", "config", "order", "missing"])
def test_reject_manifest_mismatch(addition, change):
    path, _, manifest = addition
    if change == "bytes":
        (path.parent / "rank0.pt").write_bytes(b"corrupt")
    elif change == "identity":
        manifest["pair_sha256"] = "0" * 64
        path.write_text(json.dumps(manifest))
    elif change == "missing":
        (path.parent / "rank1.pt").unlink()
    else:
        if change == "path":
            manifest["binding"]["ranks"][0]["path"] = "../original-rank0.pt"
        elif change == "config":
            manifest["binding"]["geometry"]["n_routed_experts"] = 289
        elif change == "order":
            manifest["binding"]["ranks"].reverse()
        resign(path, manifest)
    with pytest.raises((ValueError, FileNotFoundError)):
        load_pair(path)


def fake_engine():
    w = SimpleNamespace(world=2, rank=0, device="cpu", meta={},
                        cfg=SimpleNamespace(quant="exl3", hidden=4096, limit=10, streams=4),
                        layers=[])
    for _ in range(45):
        w.layers.append(SimpleNamespace(moe=SimpleNamespace(shared=True)))
    return SimpleNamespace(w=w, _gather_ints=lambda values: [values, values])


def test_startup_before_graph_capture_and_pair_identity(addition, monkeypatch, capsys):
    path, config, manifest = addition
    monkeypatch.setenv(runtime.ENV, str(path))
    engine = fake_engine()
    runtime.attach(engine, config.parent)
    assert engine.w.meta[runtime.KEY]["pair_sha256"] == manifest["pair_sha256"]
    assert json.loads(capsys.readouterr().out)["learning"] is False
    with pytest.raises(ValueError, match="reject"):
        runtime.attach(engine, config.parent)


@pytest.mark.parametrize("failure", ["config", "pair", "world", "quant", "hidden", "clamp", "shared", "missing"])
def test_both_ranks_exchange_even_on_local_startup_failure(addition, monkeypatch, failure):
    path, config, _ = addition
    monkeypatch.setenv(runtime.ENV, str(path))
    engine = fake_engine()
    calls = []

    def exchange(values):
        calls.append(values)
        other = values.copy()
        if failure == "pair":
            other[-1] ^= 1
        return [values, other]

    engine._gather_ints = exchange
    if failure == "config":
        config.write_text(config.read_text() + "\n")
    elif failure == "world":
        engine.w.world = 1
    elif failure == "quant":
        engine.w.cfg.quant = "tr3"
    elif failure == "hidden":
        engine.w.cfg.hidden = 4095
    elif failure == "clamp":
        engine.w.cfg.limit = 9
    elif failure == "shared":
        engine.w.layers[11].moe.shared = None
    elif failure == "missing":
        monkeypatch.delenv(runtime.ENV)
    with pytest.raises(ValueError, match="reject"):
        runtime.attach(engine, config.parent)
    assert len(calls) == 1
    assert runtime.KEY not in engine.w.meta


def test_reject_different_model_geometry():
    with pytest.raises(ValueError):
        geometry({"hidden_size": 12})


def test_existing_output_never_overwritten(addition):
    path, config, _ = addition
    with pytest.raises(FileExistsError):
        bundle(path.parent / "rank0.pt", path.parent / "rank1.pt", config, path.parent)
