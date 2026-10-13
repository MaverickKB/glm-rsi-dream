"""Protect the stock routed-only format policy without running a converter."""

from types import SimpleNamespace

import pytest

from glm_rsi_dream.exl3_recipe import recipe_for_modules


def linear(key, budget="bits", qmap="block.mlp"):
    return SimpleNamespace(key=key, qbits_key=budget, qmap=qmap, modules=[])


def test_routed_only_recipe_preserves_always_active_matrices():
    prefix = "model.language_model.layers.11."
    modules = []
    for expert in (0, 287):
        for projection in ("gate", "up", "down"):
            modules.append(linear(prefix + f"mlp.experts.{expert}.{projection}_proj"))
    for projection in ("gate", "up", "down"):
        modules.append(linear(prefix + f"mlp.shared_experts.{projection}_proj"))
    modules.append(linear(prefix + "self_attn.q_a_proj"))
    modules.append(linear("model.language_model.layers.0.mlp.gate_proj"))
    modules.append(linear("lm_head", "head_bits"))
    modules.append(linear("model.language_model.layers.45.mlp.experts.0.gate_proj", "mtp_bits"))
    modules.append(linear(prefix + "mlp.gate", qmap=None))
    parent = SimpleNamespace(modules=modules)
    recipe = recipe_for_modules([parent])
    assert recipe["head_bits"] == 16
    assert len(recipe["tensors"]) == 11
    assert recipe["tensors"][prefix + "mlp.experts.287.down_proj"] == 4
    assert recipe["tensors"][prefix + "mlp.shared_experts.down_proj"] == 16
    assert recipe["tensors"][prefix + "self_attn.q_a_proj"] == 16
    assert recipe["tensors"]["model.language_model.layers.0.mlp.gate_proj"] == 16
    assert "lm_head" not in recipe["tensors"]
    assert prefix + "mlp.gate" not in recipe["tensors"]


def test_missing_routed_matrices_and_duplicate_descriptors_fail():
    with pytest.raises(ValueError, match="no GLM routed"):
        recipe_for_modules([linear("model.language_model.layers.3.mlp.shared_experts.gate_proj")])
    routed = linear("model.language_model.layers.3.mlp.experts.0.gate_proj")
    with pytest.raises(ValueError, match="Duplicate"):
        recipe_for_modules([routed, routed])
