"""Prepare a routed-only EXL3 conversion recipe from the upstream model layout."""

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROUTED = re.compile(r"\.mlp\.experts\.\d+\.(gate|up|down)_proj$")


def recipe_for_modules(modules: Any) -> Any:
    """Use upstream budgeted-linear descriptors. Keep shared/attention/dense at 16 bits."""
    pending = list(modules)
    tensors = {}
    routed_count = 0
    while pending:
        module = pending.pop()
        pending.extend(module.modules)
        if getattr(module, "qmap", None) is None or getattr(module, "qbits_key", None) != "bits":
            continue
        key = module.key
        if key in tensors:
            raise ValueError(f"Duplicate upstream budgeted tensor {key}")
        if ROUTED.search(key):
            tensors[key] = 4
            routed_count += 1
        else:
            tensors[key] = 16
    if not routed_count:
        raise ValueError("Upstream layout exposes no GLM routed expert matrices")
    return {"target_bpw": 4, "head_bits": 16, "tensors": tensors}


def prepare_recipe(checkpoint: Any, output: Any) -> Any:
    # Import the external converter only when explicitly preparing its recipe.
    from exllamav3.conversion.allocation import create_q_strategy_from_recipe
    from exllamav3.model import Config, Model

    output = Path(output)
    if output.exists():
        raise ValueError("Recipe output already exists")
    raw = json.loads((Path(checkpoint) / "config.json").read_text())
    text = raw.get("text_config", raw)
    if text.get("model_type") != "glm5_next_text":
        raise ValueError("Only GLM-5.3-Flash is in scope")
    if raw.get("quantization") or raw.get("quantization_config"):
        raise ValueError("Fresh conversion requires a floating source; retain routed EXL3 bytes for mixed input")
    if text.get("num_nextn_predict_layers", 0):
        raise ValueError("Stock converter has uniform MTP precision; routed-only MTP remains unqualified. "
                         "Use the existing routed-only EXL3 checkpoint as export input.")
    config = Config.from_directory(str(checkpoint))
    model = Model.from_config(config)
    mtp = None
    if "mtp" in config.model_classes:
        mtp = Model.from_config(config, component="mtp")
    recipe = recipe_for_modules(model.modules)
    # Let the actual upstream allocator verify exact coverage before publishing.
    create_q_strategy_from_recipe(model, mtp, config, recipe["tensors"], 16, 16)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(recipe, indent=2) + "\n")
    return recipe


def main() -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recipe = prepare_recipe(args.checkpoint, args.output)
    print(json.dumps({"recipe": str(args.output), "budgeted_tensors": len(recipe["tensors"])}))


if __name__ == "__main__":
    main()
