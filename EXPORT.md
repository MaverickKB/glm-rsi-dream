# Shared-expert checkpoint export

Under development. This exporter processes supplied numerical additions. CPU
export checks do not establish factual learning or a delivered dream-to-neurons
path. Native end-to-end consolidation remains unverified.

`glm-dream-export` bakes the two rank additions from DREAM-neurons into standard
GLM-5.3-Flash safetensors. The output contains checkpoint tensors, configuration,
static tokenizer/processor assets and an export receipt. Inference reads the
ordinary shared-expert matrices.

## Merge contract

Inputs are a complete checkpoint with floating shared weights and the ordered
rank-0/rank-1 `shared_ffn_swiglu_v1` artifacts. Hidden size, original intermediate
width, layers, MTP presence and activation limit come from configuration. The
artifacts supply each added width. Training artifacts currently imply limit 10
unless they supply `swiglu_limit` explicitly. The exporter checks that it agrees
with the checkpoint.

For each rank and projection, concatenate its original neurons, added neurons
and zero padding. Gate/up concatenate rows; down concatenates columns. Each rank
keeps its original base neurons. The two native rank blocks then form the full
checkpoint matrix. Padding has exactly zero gate/up/down weights. Every shared
site, including MTP, receives the same global width, and unselected sites retain
their original function through zero padding.

The supported stock configuration defines shared width as
`moe_intermediate_size * n_shared_experts`. The exporter leaves the routed
intermediate width and routed count intact and increases that width multiplier
to the smallest compatible integer. With 2048 original neurons and 32 or 128 added
neurons per rank, the exported width is 4096 and the multiplier is 2. Upstream
TensorFold and Transformers each construct one fused always-active shared MLP
from those matrices. This changes the config's shared-width multiplier from 1
to 2; it does not create another routed branch. Arbitrary independently trained
nonlinear neurons generally require increased width. This implementation uses
widening.

The output is assembled in a sibling staging directory, read back with
safetensors, then published by directory rename. Existing output directories are
rejected. Source files remain intact. Unchanged tensor payloads stream verbatim,
including existing routed EXL3 packs. The receipt records input config and rank
artifact hashes, source/output shard hashes, width changes and maximum weight
rounding error. Artifact-to-base pairing remains caller-supplied because the
current training artifact has no base-checkpoint identity field.

## Commands

Install the export extra in a Python 3.11 or 3.12 environment:

```sh
pip install -e '.[export]'
glm-dream-export --base /CHECKPOINT \
  --rank0 /ADDITION/rank0.pt --rank1 /ADDITION/rank1.pt \
  --output /FRESH_MERGED_CHECKPOINT
```

The equivalent module invocation is `python -m glm_rsi_dream.export`.
Host addresses and private artifacts belong in operator configuration, outside
this source checkout.

## EXL3 paths and current limits

The reviewed stock TensorFold EXL3 loader expects EXL3 routed experts and
floating non-routed matrices, including the shared expert. The actual lane's
existing routed-only EXL3 checkpoint therefore supplies a direct path: preserve
its routed/MTP EXL3 bytes and quantization metadata, replace the shared BF16
matrices, and write a plain checkpoint. No new encoding of unchanged routed
weights is needed. The exporter records fresh quantization as false. A mixed
format regression fixture tests preservation of metadata and routed payloads;
it is a format test rather than a real EXL3 encoding or loading proof.

Fresh conversion from a floating checkpoint requires a routed-only recipe.
`python -m glm_rsi_dream.exl3_recipe --checkpoint /MERGED_FLOATING_MODEL
--output /FRESH_RECIPE.json` prepares one using the external ExLlamaV3 layout
and asks its allocator to validate exact coverage. Routed matrices get 4 bits;
shared, attention and dense matrices get 16 bits. Head, embeddings and vision
stay 16 bits. JSON is accepted by the converter's YAML recipe reader.

For a floating GLM layout with no MTP component, the prepared command is:

```sh
python /EXLLAMAV3/convert.py -i /MERGED_FLOATING_MODEL \
  -o /FRESH_EXL3_MODEL -w /FRESH_QUANT_WORK \
  --recipe /FRESH_RECIPE.json --head_bits 16 --mtp_bits 16 \
  --vision_bits 16 --embed_bits 16 --codebook mcg
```

The converter and recipe preparation were source-reviewed and policy-tested,
but have not been executed against the external ExLlamaV3 model allocator or
CUDA quantizer in this lane. The pinned GLM checkpoint has one MTP component.
The reviewed converter assigns uniform precision to that component, while the
stock TensorFold MTP path requires EXL3 routed matrices and floating other
matrices. Fresh conversion is explicitly blocked for that configuration until
a compatible per-tensor MTP conversion is qualified. The existing routed-only
EXL3 input path preserves its already encoded MTP and remains available.

The reviewed converter pin is
`3a513ac9f91aae777f317ca2341b8e445b62c64e`. Stock TensorFold source is read from
`c4646171139ee8a3c38103eaa1699dad226ec12b`. These are source-review identities,
not an assertion about the current serving containers. Exact source URLs and
hashes are in the job's `evidence/sources.json`. The width and mixed-format
contract comes from [the pinned stock loader](https://github.com/ashhart/TensorFold/blob/c4646171139ee8a3c38103eaa1699dad226ec12b/src/tensorfold/families/glm5_next/cuda/weights.py).
The conversion policy comes from [the pinned recipe allocator](https://github.com/turboderp-org/exllamav3/blob/3a513ac9f91aae777f317ca2341b8e445b62c64e/exllamav3/conversion/allocation.py).

## Verification and native acceptance

Focused CPU tests cover numerical equivalence through both clamps, rank order,
multiple sites, MTP padding, separate input shards, routed-byte preservation,
BF16 rounding, invalid artifacts/indexes and interrupted-output cleanup.
`tools/prove_export_cpu.py` uses the actual DREAM-neurons fitter, the unmodified
stock TensorFold byte splitter, its exact Config class and the stock
Transformers shared MLP. It also checks the 4096/2048 geometry on CPU.
`tools/prove_export_full_cpu.py` saves a complete tiny BF16 GLM, exports the
trained addition, reloads the standalone checkpoint through the stock
Transformers class and decodes token IDs. That model is synthetic and has no
semantic recall qualification.

The stock Mia TensorFold proof requires the GPU owner's grant and the
BRIEF-required approval for a live restart. It must identify both ranks, the
exact stock source/image/config and the exported checkpoint; establish that
inference loads no dream hooks; then record real head-generated recall and
control replies. BF16 casting, changed reduction order and the recipe's default
dense Q4 conversion can change results, so CPU float32 parity does not qualify
native reply preservation. Worker generation remains absent and TP=2 stays
intact. Capture the original pair and protected probes before any approved
cutover, then verify the same direct user path after restoring it.

Only GLM-5.3-Flash geometry and a synthetic tiny GLM layout were tested.

The export/test environment uses the package's pinned Torch 2.9.0 and
tokenizers 0.22.1. Stock Transformers 5.19.0 validation uses a separate temporary
environment because that consumer requires tokenizers 0.23.x. Transformers is
validation tooling rather than an exporter runtime dependency.
