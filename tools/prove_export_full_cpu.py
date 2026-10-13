"""Load and decode a complete tiny stock Transformers GLM checkpoint on CPU."""

import argparse
import json
from pathlib import Path

import torch
from transformers.models.glm5_next.configuration_glm5_next import (
    Glm5NextConfig,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextForConditionalGeneration,
)

from glm_rsi_dream.export import export_checkpoint


def prove(component, output):
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(929)
    text = json.loads((component / "base/config.json").read_text())["text_config"]
    text.update(intermediate_size=16, vocab_size=32, num_attention_heads=2,
                num_key_value_heads=2, q_lora_rank=8, kv_lora_rank=8,
                qk_nope_head_dim=8, qk_rope_head_dim=0, v_head_dim=8,
                index_n_heads=2, index_head_dim=8, index_topk=4, index_kpool=1,
                num_experts_per_tok=2, n_group=1, topk_group=1,
                eos_token_id=[28], bos_token_id=1, pad_token_id=0)
    vision = Glm5NextVisionConfig(depth=1, hidden_size=16, intermediate_size=16,
                                num_heads=2, out_hidden_size=8, patch_size=2,
                                spatial_merge_size=1)
    config = Glm5NextConfig(text_config=Glm5NextTextConfig(**text), vision_config=vision,
                          image_token_id=31, image_start_token_id=30, image_end_token_id=29)
    model = Glm5NextForConditionalGeneration(config).to(torch.bfloat16).eval()
    base = output / "base"
    model.save_pretrained(base)
    artifacts = [component / "rank0.pt", component / "rank1.pt"]
    merged = output / "merged"
    report = export_checkpoint(base, artifacts, merged)
    del model
    # This is a stock class loading the standalone exported checkpoint with no addon hooks.
    loaded = Glm5NextForConditionalGeneration.from_pretrained(merged, dtype=torch.bfloat16).eval()
    prompt = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        logits = loaded(input_ids=prompt).logits
        if not torch.isfinite(logits).all():
            raise ValueError("Tiny stock model produced nonfinite logits")
        tokens = loaded.generate(prompt, max_new_tokens=3, do_sample=False,
                                 eos_token_id=28, pad_token_id=0)
    result = {
        "status": "FULL TINY FLOATING CHECKPOINT CPU PASS", "device": "cpu",
        "stock_class": type(loaded).__name__, "full_tiny_checkpoint": True,
        "generated_token_ids": tokens.tolist(), "finite_logits": True,
        "merged_shared_width": loaded.model.language_model.layers[3].mlp.shared_experts.intermediate_size,
        "merge": report, "semantic_recall_measured": False,
        "exl3_conversion_executed": False, "stock_mia_tensorfold_answer_verified": False,
    }
    (output / "proof.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prove(args.component, args.output)


if __name__ == "__main__":
    main()
