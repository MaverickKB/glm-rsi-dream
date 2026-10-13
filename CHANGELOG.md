# Changelog

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
