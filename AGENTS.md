# Coding contract

GLM RSI Dream is an experimental two-node NVIDIA DGX Spark GB10 attention-memory
loop using TP=2. Preserve the capture, real 96-row gradients at all 11
full-attention sites, generated-answer checking, and keep/reject path.

The status is v0.0.1-preview. End-to-end Spark execution as packaged and
ordinary-chat serving/rollback acceptance remain in progress. CPU dry runs
must open no process or network connection and create no run directory. Live
execution must be explicitly requested and use the configured pair.

The head Spark coordinates orchestration and grading; both ranks own capture
and fitting. Generation uses only the head API. Verify that the worker has no
generation listener. Host addresses, SSH identity, paths, and ports belong in
ignored config.json. Private facts and tensors belong in private run directories.
Use only fictional examples in source. Publish aggregate scores without private
run receipts or private-data fingerprints.

The packaged resources/manifest.json under glm_rsi_dream owns runtime/source
pins, model identity, method settings, and preview status. Startup validation
must use that manifest and reject mismatched sources. Preserve original engine
files, retained Apache-2.0 headers, LICENSE, and NOTICE. Record borrowed-code
provenance in UPSTREAM.md. Model weights and containers have separate terms.

Write linear Python for a junior reader, with named intermediates, loops, and
clear if/else branches. Keep changes small. Use ordinary punctuation without
em dashes. Do not add hype or unsupported outcome claims.

From the checkout, use Python 3.11 or 3.12:

```sh
pip install -e '.[dev]'
ruff check .
CUDA_VISIBLE_DEVICES='' python -m pytest -q
python -m build
glm-rsi-dream --config config.example.json --dry-run run
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
