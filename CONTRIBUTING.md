# Contributing

Open an issue to describe a substantial behavior change before implementing it.
Keep pull requests small and focused. Explain the problem, the changed behavior,
and the checks you ran. Write linear Python with named steps and readable loops.

Supported Python versions are 3.11, 3.12, and 3.13. Run the CPU checks before submitting.
Dev dependencies are locked in `uv.lock`. Install them with uv 0.13 or newer:

```sh
uv sync --locked --extra dev
uv run ruff check .
uv run mypy
CUDA_VISIBLE_DEVICES='' uv run python -m pytest -q
```

CI runs Ruff, mypy, the CPU tests, and the wheel install on Python 3.11, 3.12,
and 3.13. CPU tests also run on macOS. After changing a direct pin, refresh
the lock with `uv lock` and commit `uv.lock`.

Tests marked `gpu` skip when CUDA is unavailable. CPU checks do not establish
native Spark acceptance. Report live evidence separately, with the runtime
identity, a fresh held-out set, generated-answer grading, control preservation,
and exact restoration. Never train or select against a consumed holdout.

Submit only public source and fictional examples. Exclude personal identifiers,
credentials, internal addresses, local configuration, logs, captured tensors,
private prompts, private-data fingerprints, and run directories. Tensor states
can encode teaching facts. Scan the complete tree, including hidden and
compressed files, before publication.

By submitting a contribution, you license your original package changes under
the repository's MIT license. Changes to Apache-2.0 integration or fixture files
remain Apache-2.0 with their existing notices. Documentation with an explicit
upstream license retains that license. You must have the rights to contribute
the material and preserve required attribution. Borrowed code must use MIT,
Apache-2.0, BSD, or public-domain terms, with provenance recorded in UPSTREAM.md.
Flag other terms for review before adding the material.

No DCO sign-off or contributor license agreement is required.
Follow [the code of conduct](CODE_OF_CONDUCT.md). Use
[the security reporting process](SECURITY.md) for sensitive reports.
