"""Same v2 bytes through stock FFN hooks and the owner's four-stream update."""

import hashlib
import json

import pytest
import torch
from dream_loader import runtime
from dream_loader.artifact import FOUR_STREAMS, identity, load_pair
from dream_loader.bundle import bundle
from four_stream_probe import parity_case
from test_validation import fake_engine


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("rows,prefill,chunked,mode", [(1, False, False, "1"),
                                                      (7, False, True, "0"), (65, True, True, "1")])
def test_four_stream_stock_ffn_matches_owner(four_stream_addition, monkeypatch, rank,
                                            rows, prefill, chunked, mode):
    path, _, _ = four_stream_addition
    for index in (11, 23, 35, 19):
        parity_case(path, rank, rows, prefill, chunked, mode, index)


def test_v2_round_trip_and_startup(four_stream_addition, monkeypatch):
    path, config, manifest = four_stream_addition
    assert manifest["binding"]["mechanism"] == FOUR_STREAMS
    for rank in range(2):
        original = path.parent.parent / f"original-rank{rank}.pt"
        assert original.read_bytes() == (path.parent / f"rank{rank}.pt").read_bytes()
    monkeypatch.setenv(runtime.ENV, str(path))
    engine = fake_engine()
    runtime.attach(engine, config.parent)
    assert runtime.four_streams(engine.w) is True
    engine = fake_engine()
    engine.w.cfg.streams = 1
    with pytest.raises(ValueError, match="reject"):
        runtime.attach(engine, config.parent)


def test_four_stream_zero_down_preserves_stock_base(four_stream_addition):
    path, config, _ = four_stream_addition
    originals = []
    for rank in range(2):
        source = path.parent / f"rank{rank}.pt"
        value = torch.load(source, weights_only=True)
        for site in value["layers"].values():
            site["down"].zero_()
        output = path.parent.parent / f"zero-rank{rank}.pt"
        torch.save(value, output)
        originals.append(output)
    output = path.parent.parent / "zero-addition"
    bundle(originals[0], originals[1], config, output)
    for rank in range(2):
        parity_case(output / "manifest.json", rank, 7, True, True, "0", 35, expect_change=False)


@pytest.mark.parametrize("change", ["one_stream_down", "three_stream_down", "mixed_rank", "binding"])
def test_v2_rejects_wrong_stream_geometry(four_stream_addition, change):
    path, _, manifest = four_stream_addition
    rank_file = path.parent / "rank0.pt"
    value = torch.load(rank_file, weights_only=True)
    if change == "binding":
        manifest["binding"]["mechanism"] = "shared_ffn_swiglu_v1"
    else:
        if change in ("one_stream_down", "three_stream_down"):
            streams = 1 if change == "one_stream_down" else 3
            value["layers"][11]["down"] = value["layers"][11]["down"][:streams * 4096].clone()
        elif change == "mixed_rank":
            value["mechanism"] = "shared_ffn_swiglu_v1"
            for site in value["layers"].values():
                site["down"] = site["down"][:4096].clone()
        torch.save(value, rank_file)
        manifest["binding"]["ranks"][0]["sha256"] = hashlib.sha256(rank_file.read_bytes()).hexdigest()
    manifest["pair_sha256"] = identity(manifest["binding"])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_pair(path)
