# Changelog

## Unreleased

- Add the RSI Dreambox portal, locked host sessions, local API, prepared intake,
  configurable extraction/grading, and a disabled-by-default traffic tap.
- Integrate experimental dream lifecycle, neuron research, knowledge revision,
  export, and inference-loader lanes. Shared-expert dream learning remains
  Under development, with new dream and schedule actions gated off by default.
- Use portable operator configuration and verify both distributions on CPU.

- Run Ruff, strict mypy, Hypothesis checks, and CPU coverage on Python 3.11, 3.12, and 3.13.
- Reject admission geometry that does not fit the signed int32 rank exchange, and release only that request. An overflowing field no longer marks the pair broken.
- Compare memory context limits using the one generated token native admission substitutes for a nonpositive count. Non-integer counts fail closed.
- Record a request-local admission error when lane, capacity, or length checks fail, instead of escaping into the pair-wide breaker.
- Reject non-finite manifest JSON, malformed manifest structure, path escape, and malformed candidate digests.
- Finish rollback as restored or restore_failed when a candidate record is malformed. A failed restore leaves the changed flag set so a later window retries, and the phase does not stay on restoring.
- Reject a staged archive member whose contents cannot be read.
- Refuse an overlay module whose origin is not a filesystem path.
- Raise the existing nonfinite-gradient error when a fitting gradient is missing.

## v0.0.2

- Lock dev dependencies and run CPU checks on Python 3.11, 3.12, and 3.13 for Linux and macOS. Package metadata requires `>=3.11,<3.14`. Pushing a version tag attaches the sdist, wheel, and checksums to the GitHub release and does not publish to PyPI.
- Reserve attention-memory prompt extents before native request admission, including concurrent and long-context requests. Requested output uses native extent growth.
- Agree memory preparation and local restoration across both TP ranks before forward execution. Failed installation releases only its request.
- Report rank desynchronization in health and return HTTP503 for an unhealthy pair.
- Add CPU regressions for reservation, paired failures and health responses.
- Release a memory slot and clear its attention rows when the rank exchange fails after a local install.
- Parse a valid eval tag even when this rank's capture observer is disabled.
- Treat an explicit unhealthy or rank-desynchronized health body as a failed probe, including HTTP 503.
- Reject a non-positive guard sample interval and a backwards swap counter before dividing.
- Keep a failure observed during the restore probe.
- Grade the helper fact with the alias that appears in its statement.
- When rank 1's follower hits an error during eval admission, it joins the peer's admission vote and drops its own slot instead of leaving that vote blocked.
- Report cache writes and the off-path delta as separate receipt fields, and reject a control when either one is nonzero.
