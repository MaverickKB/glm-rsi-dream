# GLM DREAM inference loader

Experimental inference support for supplied numerical additions. Shared-expert
dream learning is Under development. CPU numerical parity does not establish
native factual recall or a delivered consolidation path.

This separate wheel loads the DREAM shared-expert addition into the stock
TensorFold EXL3 GLM path. It uses `shared_ffn_swiglu_v1` and
`shared_ffn_four_streams_v2` rank artifacts without changing their bytes.
Every token at an included layer receives the added clamped SwiGLU correction.
V1 adds one shared branch before routed/shared combine. V2 reduces four
independent corrections across ranks and writes each through its HC post coefficient.
The base expert and routed-expert count stay unchanged.

The supported target is GLM-5.3-Flash, hidden 4096, shared intermediate 2048,
one shared expert, 288 routed experts, 45 layers and three initial dense layers.
The source profile is the exact `pins.json` set already retained by LOOP2,
including stock engine SHA `47513241...` and forward SHA `f54b7f76...`.
[Mia's reviewed recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/3fd60dfc836fab0f1ed0fd4f0afaf2a1d62b2aab)
launches two ranks through the `tensorfold` command. Source versions with other
hashes require a separately reviewed and tested profile.

The wheel contains only inference, artifact validation, bundling, import
bootstrap, and serving delegation. PyTorch and TensorFold come from the stock
image. Attention pockets are optional in the brief and are outside this
loader's current scope. It has no optimizer, capture, request routing, training,
or dream trigger. The loader consumes a fixed addition for the lifetime of a
fresh serving process. It does not modify stock engine files on disk.

## Build and inspect offline

Run from this directory with Python 3.11 or 3.12 and build/setuptools installed:

```sh
python -m build
python -m pip install --no-deps dist/glm_dream_loader-0.2.0-py3-none-any.whl
glm-dream-loader bundle --rank0 /path/rank0.pt --rank1 /path/rank1.pt \
  --model-config /path/base-model/config.json --output addition
glm-dream-loader inspect addition/manifest.json
```

The input files are exactly the neuron trainer's schema version 1 artifacts:
rank 0 and 1, world 2, frozen base, middle layer indices 3 through 39, and finite
FP32 `gate[width,4096]`, `up[width,4096]`, `down[4096,width]` matrices for V1.
V2 uses `down[16384,width]`, flattened in stream-major order, with the same
schema version 1 and other fields. Both ranks
must contain the same placements and widths. Width is read from tensors.
Bundling copies the existing bytes and creates a paired SHA-256 identity.
A manifest binds those files and exact model-config bytes. It does not certify
the base checkpoint's weight bytes or the addition's factual accuracy. Use the
owner's retained checkpoint and acceptance evidence for those identities.

## Stock recipe integration

Build the derived image locally from this directory after creating `addition/`.
The Dockerfile packages the wheel and fixed artifacts in the image, so the
stock recipe's existing mount and environment propagation remain sufficient:

```sh
docker build -t glm-dream-loader:0.2.0 .
```

Make the same image available to both Sparks through your existing image
transport. The build's pinned base image is listed in `pins.json`. Configure
Mia's existing recipe with `IMAGE=glm-dream-loader:0.2.0`, retaining its current
TP=2, model/drafter paths, fabric, port, and host configuration. The derived
image's `tensorfold` command invokes `dream_loader.serve`, which installs the
import hooks in two source-pinned modules and calls the original installed TensorFold
console entry point with unchanged arguments. Rank 1 receives the recipe's
rank-1 arguments without an API host or port. Rank 0 owns generation.

An already installed extension can instead run `glm-dream-serve serve` with
those same stock TensorFold arguments and `TF_GLM_DREAM_MANIFEST` set on each
rank. Both rank files must exist beside the manifest on both hosts. For this
job, GPU/serving use needs a DREAM-neurons queue grant, and a live restart needs
owner approval. The Dockerfile and this document authorize no
live deployment.

The extension loads after stock weight loading and before draft engine/main
engine graph capture. Every startup checks source hashes, artifact bytes,
rank ownership, model config, geometry, clamp 10 and existing shared sites.
The stock `_gather_ints` exchange compares paired addition identity. A local
validation failure still enters that exchange; both ranks reject startup.
There is no fallback to serving the base model while claiming an addition.
Each successful rank prints its pair hash, mechanism, layer list and inference-only mode.

V2 retains the four BF16 residual streams before the FFN. It computes a FP32
correction per stream, gathers both rank shards, and sums them in rank order.
The exchange sends contiguous flattened buffers through stock `all_gather`.
For each stream, it casts `base FFN + correction` to the residual dtype before
applying that stream's frozen post coefficient. Frozen residual mixing uses
`combine[source,destination]`. The resulting residual is cast back to BF16.
This follows the owner's native four-stream arithmetic and rounding order.
It retains the base FFN, shared and routed expert counts, and attention paths.

The stock HC row-split prompt path performs residual mixing inside its exchange
and bypasses the regular FFN hooks. V2 selects the stock unsplit prompt path at
both normal and layer-sliced dispatch points. V1 retains the stock row-split
choice. CUDA graph capture/replay and performance of V2 remain untested.
The spec's native promotion checks also require owner base/study identity and
clean sequence evidence. This standalone manifest binds config and artifact
bytes. It does not certify the study, every base weight byte or native acceptance.

Rollback is the preserved original image and original serving configuration,
restarted as the existing pair under the owner's approved recovery procedure.
This loader does not implement service management. A new bundle is loaded only
in fresh paired processes, keeping the model's retained prompt states and
captured graphs tied to one fixed addition.

## Verification and limits

From the owning checkout:

```sh
CUDA_VISIBLE_DEVICES='' python -m pytest -q inference-loader/tests \
  -c inference-loader/pyproject.toml
ruff check inference-loader/src inference-loader/tests
```

The CPU parity test executes the whole pinned stock `moe_block` function with
explicit CPU doubles for CUDA routing, base shared computation and reduction.
Its loader insertion is compared to the frozen DREAM native `shared()` function
from dream-neurons commit `e198c2d83d05759734afeb0793e9257873f57045`. Both consume
the exact same saved rank files. Production hidden dimension 4096, width 32,
placements 11/23/35, an untouched layer, BF16 inputs, FP32 outputs, rank 0/1,
1/7/65 rows, chunked fills, and `TF_GLM_COMBINE_SY=0/1` are covered. Paired summed
corrections also match exactly. The test is numerical inference-extension proof,
not full-model GLM answer or learned-memory acceptance.

The V2 tests execute the entire FFN residual update at hidden4096, width32 per
rank, sites11/23/35 and untouched19, both ranks, and1/7/65 rows. Independent
rank tensors, asymmetric stream writes, BF16 residuals, FP32 post/comb, chunked
fills and both combine paths are covered. The exact owner reference is retained
in `tests/dream_four_stream_reference.txt`. The offline tool
`python tests/four_stream_probe.py addition/manifest.json` emits a JSON receipt.
Inside the stock image, add `--stock-imported` to run its actual imported
`layer_forward`, `_ffn` and `moe_block` with explicit CPU backend doubles.

CUDA kernels, NCCL, graph capture/replay, performance, full-model outputs,
full-model Docker serving startup on the Sparks and learned recall have not
been run in this lane. A derived image was built on the head Spark without
network access, GPUs, model loading or service changes. All 22 source pins
validated and both stock modules imported through the extension. The V1 MoE
probe and V2 full FFN residual-update probe each matched their frozen native
reference exactly in 24 CPU cases inside their respective derived images.
The synthetic additions do not encode learned facts. The intended model
is GLM-5.3-Flash; no other model is supported or qualified. No live service has
been changed and no public push has occurred.

The source archive is also self-contained for CPU parity tests. From the
unpacked archive, run `python -m pytest -q` after installing pytest and PyTorch.
`tests/stock_image_parity.py` repeats the comparison in the actual derived
stock image, using its imported MoE function and explicit CPU backend doubles.
Its arguments are the paired manifest path and frozen native reference path.
For an owner-fitted candidate, it also accepts a verified native reference hash
and a paired capture manifest:

```sh
python tests/stock_image_parity.py addition/manifest.json native.py \
  --reference-sha256 VERIFIED_NATIVE_SOURCE_SHA256 \
  --capture-manifest capture-manifest.json \
  --artifact-origin 'owner-fitted diagnostic candidate'
```

The capture manifest contains a `ranks` list with rank, path and SHA-256 for
each native student capture. The runner checks those bytes, TP=2 rank ownership
and `attention_memory=False`, then consumes the saved BF16 FFN inputs. Layer
placements and widths come from the actual addition. Routing, base shared
output, combine and reduction still use explicit CPU doubles.

This probe passed for all eight DREAM-neurons study01 fitted configurations:
single placements 11/23/35 and combined 11/23/35, each at width 32 or 128 per
rank. All 192 cases match exactly against native source from commit `47559c4`,
with nonzero corrections at every included site. This is fitted-artifact
numerical fidelity. The owner retains placement selection and learned-recall
acceptance.
