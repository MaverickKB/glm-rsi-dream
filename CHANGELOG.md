# Changelog

## v0.0.2

- Lock dev dependencies and run CPU checks on Python 3.11, 3.12, and 3.13 for Linux and macOS. Pushing a version tag attaches the sdist, wheel, and checksums to the GitHub release and does not publish to PyPI.
- Reserve attention-memory prompt extents before native request admission, including concurrent and long-context requests. Requested output uses native extent growth.
- Agree memory preparation and local restoration across both TP ranks before forward execution. Failed installation releases only its request.
- Report rank desynchronization in health and return HTTP503 for an unhealthy pair.
- Add CPU regressions for reservation, paired failures and health responses.
