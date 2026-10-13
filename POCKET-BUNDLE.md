# Portable personalization folder

Shared-expert dream learning is Under development. Portable file validation and
CPU round trips do not establish native factual consolidation.

`glm-pocket-bundle` exports, verifies and imports both attention ranks and an
optional paired `shared_ffn_swiglu_v1` or `shared_ffn_four_streams_v2` addition.
It operates on CPU and writes a
fresh directory. Install the package with its `export` extra. Source artifacts
and previous imports remain available for editing and recovery.

The supported model is GLM-5.3-Flash. The native runtime, its pinned source and
the base weights are separately installed on the destination. The folder carries
config/tokenizer identity, attention state, pocket facts and routing metadata,
and optional added weights. It does not carry the full base weight checkpoint.

## Export and verify

Gather each rank's existing **pair directory**, containing `pair.json`,
`prefix.json`, `automatic.json`, its matching immutable
`automatic-REVISION.json`, and the selected/full tensors referenced for that
rank. Copy each numerical file from its owning rank and preserve its expected
hash. Both directories must contain the same catalog and pair binding.

Supply a JSON base identity with exactly these fields:

```json
{
  "model": "GLM-5.3-Flash-EXL3",
  "version": "5.3-Flash",
  "repository": "Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw",
  "revision": "25a44fdbf16862a46b7cc9921142c6c81350af2f"
}
```

That repository and revision come from DREAM-neurons' 00:52 ET owner correction
of both active serving commands and snapshot paths. Earlier packaged labels
named a different repository. Verify the destination snapshot from its actual
launch path before supplying the identity. This command verifies config and
tokenizer bytes plus the declared immutable revision. It does not establish
full base weight-byte identity or authenticate who authored a bundle.

```sh
glm-pocket-bundle export \
  --attention-rank0 /PRIVATE/rank0/pair \
  --attention-rank1 /PRIVATE/rank1/pair \
  --base /MODELS/EXACT_SNAPSHOT --base-identity /PRIVATE/base-identity.json \
  --version 1.0 --rank0 /PRIVATE/addition0.pt --rank1 /PRIVATE/addition1.pt \
  --output /PRIVATE/personalization-v1

glm-pocket-bundle verify /PRIVATE/personalization-v1
glm-pocket-bundle verify /PRIVATE/personalization-v1 \
  --base /MODELS/EXACT_SNAPSHOT --base-identity /PRIVATE/base-identity.json
```

Omit both `--rank0` and `--rank1` for an attention-only bundle. A single rank is
rejected. Export reads geometry from config and validates paired finite FP32
additions for the existing clamped SwiGLU inference path.

## Four residual streams

The v2 artifact keeps the existing schema version 1, rank ownership, frozen base,
and per-layer `gate`, `up`, `down` fields. Gate and up have shape `[width, hidden]`;
down has shape `[4 * hidden, width]`. Its four contiguous hidden-size blocks are
ordered by residual stream. Both ranks carry the same mechanism, target layers
and widths. GLM's `hc_mult` must be 4. Width and hidden size come from the tensors
and config, so every independent output block survives unchanged.

The shared manifest binds an `output_contract`: four streams, stream-major hidden
layout, rank-order sum, injection into the branch before HC post weighting, and
casting `(base + stream_delta)` to the residual dtype before that weighting.
This matches the owner's `native.py` writer. The folded shared input, clamped
SwiGLU activation, and frozen HC combine/post coefficients retain their existing
roles. Each destination stream receives its own correction. The manifest hash
covers these semantics, and verify checks them against the artifact mechanism.
Import records `shared_mechanism` and `shared_mode: native_four_streams` in
`boot.json`. Both rank payloads and native activation files remain byte identical.
V2 artifacts must declare the same valid `study_sha256` on both ranks. The shared
manifest binds that hash alongside the config hash; the outer manifest binds the
base revision and tokenizer hash. The owner supplies the study seal. These checks
preserve its identity without certifying training membership or author authority.
The write contract is reconciled with `ARCH-deepdive/SPEC-4stream.md`, sections 2
and 3, including source-to-destination combine orientation and separate rank sums
for base and correction.

Ordinary stock GLM shared FFN export has one hidden-size output vector that HC
distributes to four streams. `glm-dream-export` accepts v2 only when all four down
blocks within each rank are identical, an exact parameter mapping to that branch.
It records `four_stream_reduction: identical_down_blocks`. Checkpoint dtype
rounding and execution reduction order retain the existing qualification limits.
Unequal blocks fail before any output checkpoint is written. The exporter never
averages or drops streams. This conservative mapping rule does not try to prove
special cancellations across different rank activations. General independent
outputs travel through the portable bundle and the four-stream inference loader.

## Move and import

Copy the complete folder to the destination. Verify against that server's
base snapshot before importing:

```sh
glm-pocket-bundle import /TRANSFER/personalization-v1 \
  --base /MODELS/EXACT_SNAPSHOT --base-identity /PRIVATE/base-identity.json \
  --output /PRIVATE/import-v1
```

Import publishes `bundle/` and `boot.json` together after verification. All
payload and boot paths are relative, so the resulting folder can move again.
The manifest contains format/version, model/repository/revision, config and
tokenizer hashes, TP world, each payload hash and size, and a canonical manifest
hash. Symlinks, escaped paths, extra files, damaged tensors, inconsistent ranks,
catalog revisions and wrong target identities are rejected.

For the existing DREAM-neurons-capable attention runtime, mount each rank's
`bundle/attention/rankN` at its pair root. Set `TF_DREAM_EVAL=1` and
`TF_DREAM_EVAL_MANIFEST` to that rank's `pair.json`. The imported directory also
contains `automatic.json` and its frozen revision. Optional `neurons-current.json`
selects the local `rankN.pt` through the existing native shared hook.

The separate `bundle/shared/manifest.json` is readable by the inference-only
DREAM-loader wheel with `TF_GLM_DREAM_MANIFEST`. Use that path with the stock-only
loader recipe. The attention and stock-loader startup finders each own the same
engine import; this bundle uses the native `neurons-current.json` path for the
combined runtime. Installation and native serving qualification retain their
owner and GPU-queue approvals. `boot.json` records `native_boot_verified: false`.

## Edit, recover and share

Copy an imported pair directory into a working directory on each rank. Facts,
dev metadata, and admission rules remain ordinary JSON. When revising the
catalog, recompute the `automatic.json` revision with the existing canonical
`digest(config_without_revision)[:16]`, and write the matching immutable revision
file on both ranks. Refit or replace numerical states through their existing
learning workflow, then update their hashes and export a new bundle version.
Changing fact text alone does not teach the numerical tensors.

Import each version into a fresh directory. Earlier folders retain all original
bytes; importing an old bundle into another fresh destination recovers that
version. The command never overwrites the active service or an existing import.
Interrupted imports/export copies remove their staging directory and leave the
source intact. Switching the live service still uses its paired lifecycle.

The complete folder is the shareable artifact. To share selected pockets, create
a working copy of the catalog with that subset, preserve both ranks and their
full snapshots, update the revision, and export it. Facts, routes, prefixes and
tensors are personal data, so select the intended contents before sharing.

## CPU evidence and acceptance boundary

The focused tests exercise CLI export/verify/import, multiple pockets, both
ranks, selected and full states, the eleven dense attention sites and pooled
indexer rows, optional additions, target mismatch, corruption, malformed owners,
path escapes, symlinks, interrupted copies, revisions, edit/re-export and old
version recovery. Numerical attention outputs remain identical after relocation.

`tools/prove_pocket_bundle_cpu.py` reads the relocated bundle using the actual
attention catalog/policy reader, native shared artifact reader, and inference-only
stock loader. It records their source hashes and exact CPU tensor/correction
parity. Its sample uses fictional attention tensors and a real fitted development
addition. That is artifact and consumer portability evidence. CUDA snapshot
restoration, full server startup and semantic recall remain separate native
acceptance items, requiring a DREAM-neurons queue grant.

For a v2 bundle, the same tool executes the owner's actual `before_ffn`, `shared`
and `after_ffn` hooks plus the inference-only loader on CPU. A deterministic
communication double supplies the paired partials and checks the actual local
correction before exchange. The probe includes BF16 and FP32 residuals, clamped
inputs, chunked rows, row-varying HC coefficients, both ranks, off-policy rows,
and inactive buffer tails. Counterexamples distinguish independent stream writes
from an averaged branch; zero activations verify zero-effect writes on CPU.
This validates relocated CPU arithmetic; it does not
exercise CUDA graph capture, network communication or native server startup.
