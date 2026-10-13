import json

import pytest
import torch
from dream_loader.artifact import FOUR_STREAMS, MECHANISM
from dream_loader.bundle import bundle

torch.set_num_threads(1)


def make_addition(tmp_path, mechanism):
    config = {"text_config": {"hidden_size": 4096, "moe_intermediate_size": 2048,
                              "n_shared_experts": 1, "n_routed_experts": 288,
                              "num_hidden_layers": 45, "first_k_dense_replace": 3}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    generator = torch.Generator().manual_seed(815)
    paths = []
    streams = 4 if mechanism == FOUR_STREAMS else 1
    for rank in range(2):
        sites = {}
        for layer in (11, 23, 35):
            sites[layer] = {"gate": torch.randn(32, 4096, generator=generator) * 0.25,
                            "up": torch.randn(32, 4096, generator=generator) * 0.25,
                            "down": torch.randn(streams * 4096, 32, generator=generator) * 0.01}
        value = {"schema_version": 1, "mechanism": mechanism,
                 "rank": rank, "world": 2, "base_frozen": True, "layers": sites}
        path = tmp_path / f"original-rank{rank}.pt"
        torch.save(value, path)
        paths.append(path)
    manifest = bundle(paths[0], paths[1], config_path, tmp_path / "addition")
    return tmp_path / "addition" / "manifest.json", config_path, manifest


@pytest.fixture
def addition(tmp_path):
    return make_addition(tmp_path, MECHANISM)


@pytest.fixture
def four_stream_addition(tmp_path):
    return make_addition(tmp_path, FOUR_STREAMS)
