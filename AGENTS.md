# Coding contract

GLM RSI Dream is an experimental two-node NVIDIA DGX Spark GB10 attention-memory
loop using TP=2. Preserve the capture, real 96-row gradients at all 11
full-attention sites, generated-answer checking, and keep/reject path.

The status is v0.0.1, an experimental release. A later installed source bundle
passed the raw-fact loop, bounded ordinary-chat serving, and exact rollback.
This checkout retains its supplied-dataset loop and tagged ask path; preserve
that implementation distinction when describing native acceptance. CPU dry runs
must open no process or network connection and create no run directory. Live
execution must be explicitly requested and use the configured pair.

Final source re-acceptance preserved three ordinary-serving pockets: 60/60
exact ordinary controls, 42/42 supported development/admission questions,
6/6 user-only probes, and 8/8 complete protected replies. The concept-only
continuity pocket is a retained experimental tagged artifact, inactive in
ordinary serving pending separate qualification. Its first-turn call rate
stayed 17/18; self-graded first-action quality rose from 14/18 to 16/18.
Off-path controls were 30/30 exact; knowledge-attached controls passed 30/30
model quality grades with zero inappropriate continuity calls. Keep these
control meanings, emitted calls, executed workflows, and semantic grades distinct.
When call frequency is the primary goal, future fresh runs require positive
held-out frequency gain alongside quality and controls. Preserve the completed
experiment's original freeze, grades, and secondary-quality KEEP.

The head Spark coordinates orchestration and grading; both ranks own capture
and fitting. Generation uses only the head API. Verify that the worker has no
generation listener. Host addresses, SSH identity, paths, and ports belong in
ignored config.json. Private facts and tensors belong in private run directories.
Use only fictional examples in source. Publish aggregate scores without private
run receipts or private-data fingerprints.

The packaged resources/manifest.json under glm_rsi_dream owns runtime/source
pins, model identity, method settings, and release status. Startup validation
must use that manifest and reject mismatched sources. Preserve original engine
files, retained Apache-2.0 headers, LICENSE, and NOTICE. Record borrowed-code
provenance in UPSTREAM.md. Model weights and containers have separate terms.

Write linear Python for a junior reader, with named intermediates, loops, and
clear if/else branches. Keep changes small. Use ordinary punctuation without
em dashes. Do not add hype or unsupported outcome claims.

Supported Python versions are 3.11, 3.12, and 3.13. Dev dependencies are locked
in `uv.lock`. Install them with uv 0.13 or newer:

```sh
uv sync --locked --extra dev
uv run ruff check .
uv run mypy
CUDA_VISIBLE_DEVICES='' uv run python -m pytest -q
uv build
uv run glm-rsi-dream --config config.example.json --dry-run run
```

Mark native GPU tests with pytest.mark.gpu and keep native runtime imports
inside those tests so CPU collection succeeds. CPU tests are regression
tripwires. Generated grounded answers on a fresh sealed holdout and exact
control preservation decide learning acceptance. Never fit or select against
a consumed holdout. Model self-grades and deterministic assertions remain
separate from independently reviewed grounding.

Before changing recovery, verify failed capture, malformed grading, corrupt
tensors, STOP, coordinator loss, memory pressure, and rank failure restore the
original identities. Check protected reply bytes and enumerate owned candidates,
trainers, guards, and listeners. Keep failure evidence visible. Report CPU and
native acceptance separately. Follow CONTRIBUTING.md and SECURITY.md.
