# Bug hunt

CPU reading and tests on package 0.0.2. There is no GPU in this pass, so
native serving was not rerun. Line numbers for the pinned engine are from
`tests/upstream_sources.json.gz`. Scores in `results/` were not changed.

## Fixed, with a failing CPU test

### Failed exchange left the slot and the installed rows

`complete_admission` installs attention rows, then votes again
(`src/tensorfold_dream/request_memory.py`). A failed `gather_status` sets
`decoder.broken` and raises (`request_memory.py` `gather_status`). The old
disagreement path released the slot only after a vote returned. An exception
from the second vote skipped that path, so the lane stayed allocated and the
rows written by `install` stayed in place.

Pinned `State.reset` (`families/glm5_next/cuda/forward.py:371`) zeros the KDA
conv and recurrent state, the cursor, and the position. It does not zero `kc`,
`vc`, or `index`. `load_rows` copies into those views
(`families/glm5_next/cuda/decode.py:508`).

The rejection path now releases the slot and zeros those trees before the
error propagates. An agreed install still keeps the rows. Tests:
`test_rank_exchange_failure_releases_slot_and_clears_installed_rows` and
`test_agreed_install_keeps_attention_rows`.

The rank that observes the exception can clean its own slot. A peer blocked
inside the collective is a separate item below.

### A disabled observer dropped a valid eval tag

`capture_call` returns its default without running the operation when
containment is disabled and `force` is false
(`src/tensorfold_dream/capture_native.py`). `parse_admit` used that path. One
rank could miss `eval` on a well-formed wire while the other admitted it.

`parse_admit` now parses with `force=True`. A decode error still returns the
default record, so a bad wire stays the same on both ranks. Test:
`test_disabled_containment_still_parses_eval_wire`.

### Helper reverse answers failed a fact-faithful statement

The helper statement is "Project Linden has a helper named Orbit." Reverse
tasks required the alias bigram "project helper", which that statement does
not contain. `grade` requires every alias group
(`src/glm_rsi_dream/grader.py`). A fact-faithful answer failed the
deterministic check. Other reverse aliases are phrases that do occur in their
statements.

Only those alias strings were changed to "helper", in
`examples/synthetic/lesson.json` and `examples/synthetic/eval.json`. Prompts
were left as written. Test:
`test_fact_statements_satisfy_their_alias_groups_and_not_another_fact`.
That test also checks that one fact statement does not satisfy a different
fact's groups.

### Health treated a desynchronized body as success

`Cluster.health` accepted HTTP 200 with any nonempty body. `urllib` raises
`HTTPError` on 503 before that check, so a 503 became a generic network
error. A 200 body with `ok: false` or `rank_desynchronized: true` was
accepted. The overlay already asks `/health` to return 503 in that case
(`src/tensorfold_dream/overlay.py`). The guard used the same status-and-body
check.

Both now reject that JSON on the model health URL. Dashboard and other
protected URLs stay status and nonempty body, because those services can use
`ok` for something else. Tests: `test_health_rejects_explicit_rank_desync_body`
and the first half of `test_guard_rejects_desync_health_and_backwards_swap`.

### The guard divided a backwards swap counter

`MemoryGate` rejects a non-positive interval and a backwards swap counter
(`src/glm_rsi_dream/safety.py`). `guard.run` divided first. A zero interval
could raise `ZeroDivisionError` outside the URL handler and exit without
`guard-fired.json`. The guard now records `memory telemetry is invalid` and
does not divide. Test: the second half of
`test_guard_rejects_desync_health_and_backwards_swap`.

`agent.py` sets `ready_after` before the guard process starts, so a missing
key there is not the live path.

### The restore probe erased a new failure

The restore probe cleared `Cluster.failure` so an older failure would not
block the probe, then always wrote the saved value back. A failure recorded
during the probe was discarded. The saved value returns only when the probe
leaves `failure` unset. Test:
`test_restore_probe_keeps_a_failure_seen_during_the_probe`.

## Not changed

These need the pair, or a local change would reject controls that the code
defines as exact.

### High: rank 1 has no handler around `apply`

Pinned `MultiDecoder.follow` (`families/glm5_next/cuda/multi.py:2267`) calls
`apply` with no handler. The overlay inserts `complete_admission` into `apply`
on both ranks (`src/tensorfold_dream/overlay.py`). If that call raises on
rank 1, `follow` stops. Rank 0 can already be past the same collective with
`decoder.broken` set. The rank that sees the exception now drops its slot. A
peer blocked in the gather is not cleared by this process. That hang needs
the two-rank runtime.

### High: off-path receipts are one flag

`finished` sets `cache_writes` from `memory_activated` and sets
`offpath_delta` to `0.0` when memory is off, otherwise `None`
(`src/tensorfold_dream/pocket_native.py`). The control checks in
`src/glm_rsi_dream/loop.py` and `src/glm_rsi_dream/native_cluster.py` read
`memory_activated`, `cache_writes`, and `offpath_delta` as if they were
separate observations. They are not. Replacing the constant `0.0` would fail
the off-path comparison those checks already require. The published control
counts remain what those checks recorded. They are not three independent
measurements.

### Medium: the manifest layer list is not tied to layer kind

Attach checks `world == 2`, `parallel == 4`, and 45 layers
(`src/tensorfold_dream/eval_native.py`). It does not check that
`memory.layers` are the DSA layers in capture order. Fitting trains
`full['rows']` by enumerate index and reads captures with `packet.get(layer)`
(`src/tensorfold_dream/fit_attention.py`). Pinned `multi_prefill.py` calls
`dsa_rows` with `st.kc[st.dsa_index[layer.index]]` and passes `layer.index`.
A count mismatch fails later, when a layer has no trajectories or when
`load_rows` sees a different row count (`decode.py:511`). A permutation of
the same list could pair one layer's queries with another layer's rows. That
was not proven from the public sources.

### Medium: the fitting tail trim assumes latent indexer pools

After the 11 trained rows, fitting keeps `slot_count // 4 + 1` rows
(`src/tensorfold_dream/fit_attention.py`). Pinned `_row_views`
(`decode.py:449`) stores full `kc` rows, `vc` rows only when `vc` is not
`None`, then indexer pools of length `n // 4 + 1`. The example config sets
`TF_GLM_KV` to `fp8`. Pinned `forward.py` requires the latent cache for fp8,
and latent mode has no separate value cache. A non-latent layout would
shorten value rows. That layout was not changed here.

### Low: the reference attention scale is `0.0625`

`src/glm_rsi_dream/attention.py` multiplies scores by `0.0625`, which is
`1/sqrt(256)`. Pinned dense attention uses `qk_dim ** -0.5`
(`forward.py:681`). Fitting raises when `native_output` is missing or the
reference error is above `0.01` (`fit_attention.py`). A scale mismatch fails
the fit when that check runs. The constant was left as written.

### Low: prefix padding is 128 while the manifest alignment is 32

`src/glm_rsi_dream/native_cluster.py` pads the teacher prefix to at least 96
tokens and a multiple of 128. `memory.kda_alignment` in the packaged manifest
is 32, and this loop does not read it. Pinned `prompt_grid` allows 16, 32,
64, and 128, and the chunked default is 64 (`families/glm5_next/cuda/engine.py`).
128 is a multiple of each of those grids. The pad was not reduced.

### Low: the capture writer wait returns after 20ms

`enqueue` waits at most 20ms (`src/tensorfold_dream/capture_tap.py`). A slow
writer can still be in the write loop. Attention receipts are saved and hashed
in `finished` on the request path, not through that queue. A coordinator that
reads a capture file early fails the read or the hash check.

## Docs

`README.md` said package 0.0.1. The packaged manifest is 0.0.2. The status
line now says 0.0.2 and says the result tables were not remeasured. Numbers
in `results/RESULTS.md` and `results/source-results.json` are unchanged.
`SECURITY.md` still names v0.0.1 as the written review target and says 0.0.2
was not a new security review.

No host names or addresses were added.

## Checks

`ruff check .` passed. `CUDA_VISIBLE_DEVICES='' python -m pytest -q` passed,
63 tests. `glm-rsi-dream --config config.example.json --dry-run run` printed
the v0.0.2 dry-run plan and created no run directory. `python -m build`
produced the 0.0.2 sdist and wheel. No GPU tests were run.
