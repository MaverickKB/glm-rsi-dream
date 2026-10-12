# GLM RSI Dream

An experimental attention-memory loop for GLM on two NVIDIA DGX Spark (GB10)
nodes with tensor parallelism of two (TP=2). It captures fact-conditioned teacher
answers, fits numerical attention-memory rows with real gradients, checks
free answers, and records a keep or reject decision. Backbone weights stay frozen.

## Status

**v0.0.1-preview** (Python package version `0.0.1a1`). The packaged loop has not
yet been run end-to-end on Spark as packaged. Ordinary-chat serving and rollback
acceptance are in progress. Source experiments support investigation of the
method; they do not establish acceptance of this package.

## Quick start: CPU dry run

From this checkout, use Python 3.11 or 3.12:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
ruff check .
CUDA_VISIBLE_DEVICES='' python -m pytest -q
glm-rsi-dream --config config.example.json --dry-run run
python tools/prepare_baseline.py --config config.example.json
```

These two planning commands open no subprocess or network connection and create
no run directory. The CPU tests exercise local contracts and attention gradients.
They do not load GLM weights or test native Spark serving.

## Quick start: Spark pair

Live work requires exactly two NVIDIA DGX Spark GB10 nodes, Linux ARM64,
working SSH between the coordinator and both ranks, Docker access through the
standard Unix socket, and the external compatible TensorFold runtime and model
files described in [UPSTREAM.md](UPSTREAM.md). The native containers supply CUDA,
PyTorch, and their serving dependencies. Install `.[dev]` on the head if using the
CPU checks there. Read [MODEL-TERMS.md](MODEL-TERMS.md) before any downloads.

Copy the checkout to the head Spark. Configure your own SSH identity, containers,
head API address and port, fabric address/interface settings, and protected
service probes:

```sh
cp config.example.json config.json
# Edit config.json for this pair before using --live.
```

Every operator host, storage directory, and listener port is configured in
`config.json`. The `.example` addresses and angle-bracket fields are placeholders.
Set each `work_dir` to an absolute, writable directory owned by your operator
account. The example uses `/var/tmp/glm-rsi-dream`. Choose a private run directory,
for example under `~/.cache/glm-rsi-dream`. Keep configuration and run data out of
source control. Set dashboard and protected-service URLs to services you own on
the pair. The worker remains rank 1 and exposes no generation API.

For an existing compatible serving pair, run on the head Spark:

```sh
glm-rsi-dream --config config.json --run ~/.cache/glm-rsi-dream/synthetic-1 \
  --lesson examples/synthetic/lesson.json --eval examples/synthetic/eval.json \
  --live run
```

The supplied example is fictional. Its fixed teaching and evaluation questions
are separate from the model-generated plan4 source smoke and have no published
accuracy score. The packaged loop consumes the supplied lesson and evaluation
files; it does not generate those files for you.

For a fresh pair, first inspect the offline provisioning plan, then use
`python tools/prepare_baseline.py --config config.json --live` after reviewing
its external download terms and launch settings. Fresh provisioning and the
pinned image/checkpoint compatibility still need native qualification.

The loop trains each rank sequentially, checks development answers, closes
fitting before the held-out pass, checks controls, and restores the original
containers. A completed rejection returns exit code 2; an operation failure
returns 1. Inspect `check.json` and `restore-parity.json` inside your run directory.

A kept, restored memory can be read in another temporary serving window:

```sh
glm-rsi-dream --config config.json --run ~/.cache/glm-rsi-dream/synthetic-1 \
  --question 'What is the Linden project helper called?' --live ask
```

The `ask` command uses the package's tagged evaluation path. An ordinary untagged
chat request bypasses memory installation. Natural-language routing through an
ordinary-chat service remains an open acceptance item.

## How it works

At each of 11 full-attention layers, the trainer fits 96 shared key/value memory
rows to the attention readout of fact-conditioned teacher trajectories. The
objective is attention-output mean squared error, with frozen teacher queries
and causal suffix history. Native readout and directional-gradient checks run
before fitting. Learned rows use the runtime's FP8 cache format; recurrent KDA
state from the full teacher prefix remains an anchor.

The coordinator runs on the head Spark. Capture and training run on both native
ranks; generation and model grading use the head API. Deterministic assertion
checks and GLM semantic judgments both contribute to acceptance. The package
requires configured recall and gain thresholds, per-fact coverage, and exact
control preservation. Scope admission uses explicit route terms and request
tags. A passing test suite or lower fitting loss alone cannot accept a memory.

## Source results

These are source experiments on small sets, supplied for this preview. Model
self-grading is explicitly identified below. See [results/RESULTS.md](results/RESULTS.md).

| Experiment | Learned held-out | Baseline | Full in-context carrier | Development | Controls |
| --- | --- | --- | --- | --- | --- |
| Round 3, 96 rows on 11 full-attention layers | 34/36 grounded | 1/36 | 35/36 | 12/12 | 30/30 byte-exact |
| Self-run RSI smoke, plan4 | 18/18 | 0/18 | Not reported | 6/6 | 30/30 tagged controls exact |

For plan4, the model generated fictional facts, wrote the questions, graded its
own answers, and decided **KEEP**. Tiny sets, self-grading bias, and tagged
controls limit the conclusion. Tagged controls do not establish natural-language
routing. Ordinary-chat serving testing is pending. Neither row is an
end-to-end result for the package as shipped.

## Recovery and limits

Live windows preserve original container IDs, Config, HostConfig, and image
identity. Candidate restart is disabled. The controller monitors memory,
sustained swap-out, and head health. Host-local guards also check dashboard and
protected services and restore their original on a failed probe or the recovery
deadline. Coordinator loss can leave a window active until a guard observes a
failure or reaches that deadline. STOP requests and restoration remain explicit:

```sh
glm-rsi-dream --config config.json --run ~/.cache/glm-rsi-dream/synthetic-1 \
  --live restore
```

A `STOP` file in the run directory requests restoration during an active loop.
Real interruption, paired corruption, kept-memory reuse, fresh provisioning,
and exact reply restoration still need Spark acceptance as packaged.

This is short-context attention-output distillation with a retained recurrent
anchor. It is not full-model answer-loss training. Dense-context readout, explicit
scope terms, tiny consumed sets, quantized decoding, malformed model grading,
and self-grading bias bound its evidence. Broad generalization, many-memory
composition, and ordinary-chat memory serving remain unverified. Captured tensors
and prompts can contain private facts and belong only in private run storage.

## Glossary

- **RSI:** recursive self-improvement, used here for the bounded capture, fit,
  check, and keep/reject cycle.
- **TP=2:** one model partitioned across two ranks.
- **DSA:** the model's full-attention sites where memory rows are fitted.
- **KDA:** the model's recurrent attention state retained from the teacher prefix.
- **Carrier:** the full fact-conditioned prefix used as a comparison.
- **Grounded:** a complete supported answer without unsupported additions.
- **Tagged control:** a request carrying the package's evaluation metadata while
  its policy keeps memory off.
- **Held-out:** questions excluded from fitting and selection, consumed once.

## Contributing and licenses

Read [AGENTS.md](AGENTS.md), [CONTRIBUTING.md](CONTRIBUTING.md),
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md).
Original package code is MIT. Native integration and upstream source fixtures
retain their Apache-2.0 and legacy MIT attribution. The Contributor Covenant
has its own documentation license. External weights and containers have
separate terms. [NOTICE](NOTICE), [licenses](licenses), and
[UPSTREAM.md](UPSTREAM.md) record provenance and licensing boundaries.
