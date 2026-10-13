# Manual dream lifecycle

Under development. New dream work is closed by default through the host-owned `dream_features_enabled` flag. The dream-neurons owner must first pass the pocket-OFF recall bar. The host can then open the same development flag with `--enable-dream-features` in the API or CLI. Portal development access prepares manual plans. It is separate from native learning acceptance. Existing signed rollback and interrupted-operation recovery remain available to restore an already-authorized epoch.


The Python API is `glm_rsi_dream.dream.DreamLifecycle`. The installed CLI is
`glm-dream-trigger`, also available as `python -m glm_rsi_dream.dream`.
It uses the same private storage root and host-admin signing key as LOOP2.
Training calls `glm_rsi_dream.neurons.fit.fit_packets` for ranks 0 and 1 on CPU.
The addition contains gate/up/down tensors in the existing shared expert,
with the base frozen, hidden dimension 4096, intermediate dimension 2048,
one shared expert and 288 routed experts. This format supports GLM-5.3-Flash only.

`python -m glm_rsi_dream.predream --root ROOT dream --manual --confirm
--run-id ID --pocket POCKET --training PLAN --width WIDTH --steps STEPS`
starts the same shared-expert path. Unconfirmed `predream dream` still previews
raw staging. Raw staged facts require accepted attention artifacts before they
become dream teachers. The existing top-level supplied-dataset attention loop
has its own CLI.

## Authority

The host admin configures `PREDREAM_ADMIN_KEY_FILE` and
`PREDREAM_ADMIN_PERMISSION_FILE`. `predream admin-grant --action` grants explicit
actions. The lifecycle uses `dream_register`, `dream_start`, `dream_qualify`,
`dream_activate` and `dream_retire`. Pocket acceptance also needs the existing
`stage` action to verify its decision receipts. Old grants keep their original
actions. The host key stays mode 0600 and belongs to the operating user.

Manual and supervised dream actions require `--confirm`. Automatic permits automatic
invocation only when the signed grant explicitly includes `automatic` and the requested
action. Changing the funnel policy requires a new candidate approval. Automatic changes
how an already authorized action starts; it preserves source, evaluation, pair
and native authority checks. There is no autonomous scheduler in this slice.

Exact rollback and interrupted recovery use signed receipts saved at activation.
They remain available after grant expiry. Manual rollback requires `--confirm`.
An expired grant closes new learning and activation while keeping exact recovery
available. Keep the host key and owner driver available for recovery.

## Inputs and commands

All paths below point to private operator files. Nothing installs a service.

```sh
glm-dream-trigger --enable-dream-features --root ROOT register-pocket pocket.json --confirm
glm-dream-trigger --enable-dream-features --root ROOT start --run-id epoch1 --pocket POCKET --training training.json --confirm
glm-dream-trigger --enable-dream-features --root ROOT qualify --run-id epoch1 --suite suite.json --baseline baseline.json --report report.json --evidence-kind native --confirm
glm-dream-trigger --enable-dream-features --root ROOT --native-driver driver.json activate --run-id epoch1 --confirm
glm-dream-trigger --enable-dream-features --root ROOT --native-driver driver.json retire --run-id epoch1 --confirm
glm-dream-trigger --root ROOT --native-driver driver.json rollback --confirm
glm-dream-trigger --root ROOT --native-driver driver.json recover
glm-dream-trigger --root ROOT status
```

An accepted pocket spec contains `id`, `candidate_ids`, ordered `ranks`
(`rank`, `path`, `sha256`), `acceptance_report`, `recall_group`, and
`minimum_recall`. Each candidate must still have an accepted or staged GLM
decision, pocket destination and valid host-admin approval. The acceptance report
is an operator-owned attestation with `accepted:true` and ordered `rank_sha256`.
Its digest is retained. The trigger verifies that attestation's identity;
the attention lane owns the underlying model acceptance evidence.

Previously accepted attention pills can use the existing LOOP2
`queue import-learned` operator catalog. They retain `grade:null`, so the trigger
does not invent GLM approval. Their acceptance report also names `receipt` and
`catalog_facts` (external fact id to exact statement). Registration verifies the
original catalog hash, receipt and statements under the existing `catalog`
admin permission. This opens the same dream path for accepted legacy teachers.

A training plan contains `base_sha256`, `source_pockets` (pocket id to registration
manifest digest), and ordered `packets` (`rank`, `path`, `sha256`). Each sealed
training packet contains the same source/base identities, its rank, `world:2`,
`study_sha256`, `index_sha256` and native teacher/student rows. Its directory
contains the hash-bound `study-freeze.json`, all frozen study inputs and
`capture-index.json`. The current standalone fitter requires this sealed scope. `neurons.assemble`
carries source/base identities from the sealed capture index into its result.
It uses training rows only. The lifecycle additionally requires source-pocket
and base provenance. Width is per rank; total width is twice that value. Failed fitting retains its epoch and partial files for inspection.

Qualification consumes the owner's `pocket_off_eval.evaluate` suite, complete
baseline and consolidated report. It checks report/file seals and replays raw
requests, replies, semantic judgments and both rank readbacks. Epoch, base and
loaded neuron hashes must name this artifact. Ordinary and protected rows use
exact reply/token comparisons; leakage and recall use the provided semantic
reviewer. Every source pocket retains its accepted recall denominator and facts
in the rubric. Native qualification additionally requires 60 ordinary,
8 protected, 12 leak and 24 precision rows. The owner supplies genuine native
readbacks and semantic review. Replaying evidence verifies its consistency,
and does not create new model measurements.

Later epochs must retain every previous knowledge/control case. This trainer
fits one replacement shared-expert addition. Passing the cumulative suite is
required before it may replace the preceding addition.

## Artifact and paired state

`ROOT/dream/artifacts/ID/manifest.json` contains schema version 1, model, geometry,
epoch, frozen-base hash, source pockets and acceptance digests, parent epoch,
placement/width, rank tensor digests, training-plan digest, evaluation/baseline/
suite file digests, evidence kind and a manifest content seal. Qualification has
a separate host-admin signature binding the whole manifest. Qualified artifact
files stay immutable. Lifecycle status lives separately in `dream/state.json`.

Activation preserves the exact prior paired snapshot and catalog state, disables
every attention pocket and clears conditioned caches. Source files and slot
reservations remain until retirement. Retirement requires the qualified report,
the same active epoch/base/rank hashes, every pocket OFF and clean caches; it
then frees only the consolidated source slots. Source artifacts remain available
for provenance and rollback. One rollback restores the pre-activation pair and
catalog, including slots retired since activation. A later epoch rolls back to
the exact preceding epoch.

Successful retirement appends LOOP2 `learned` candidate receipts bound to the
qualified artifact. Later intake can use them for duplicate/update review.
Rollback restores the previous latest candidate snapshots. The queue's
append-only audit history remains. The signed operation journal includes both
candidate states, so interrupted retirement also restores its partial receipts.

`dream/operation.json` is signed and saved before a paired mutation. A partial
reload or failed readback attempts exact recovery immediately. A failed recovery
leaves a pending journal. A fresh client can call `recover`; new lifecycle work
waits for that bounded recovery path. Paired reload uses a compare-and-swap
precondition to reject a runtime changed by another owner during preparation.

## Native driver boundary

`--harness-pair FILE` uses an atomic filesystem pair with fictional CPU state.
It proves lifecycle wiring only. Spark mode rejects harness evidence and drivers.

`--native-driver SPEC` invokes the native owner's pinned Python runner. SPEC
contains `python`, `script`, `script_sha256`, `gpu_queue`, `grant_line`,
`restart_approval`, `restart_approval_sha256` and optional `timeout`.
The exact queue line begins `GRANT DREAM-trigger ` and must exist in the owner
queue. The approval JSON names `authority:"owner approval"`,
`allow_paired_reload:true`, and explicitly permitted integer `epochs`.
It includes a timezone-aware `expires_et`; new native changes require a current
receipt. Exact signed recovery uses the preserved runner after grant/receipt
expiry, without authorizing a new target.
The operator provisions the receipt from the owner's real approval. Creating an
assertion file does not establish owner authority.

The runner receives versioned JSON on stdin and returns JSON with `ok:true`:

- `snapshot`: return `snapshot` with frozen-base identity, epoch and ordered ranks.
- `prepare`: receive manifest, artifact directory and `retire`; return `target`.
  Stage both rank files and compute the complete target without a serving change.
- `guarded_paired_reload`: receive `target`, `expected`, and `recovery`.
  Apply both ranks under the acknowledged native guard. Compare `expected`
  atomically before normal changes. Recovery restores the operation's owned
  original identities, including when a candidate rank has failed.

Snapshots retain exact container, launch-config and image identities. Each rank
has integer `rank`, `world:2`, `running:true`, head-only `generation_api`,
`container_id`, `config_sha256`, `image_sha256`, `neurons_sha256`, `pockets`,
`active_memory_rows` and `pocket_cache_clean`. Each pocket has `enabled` plus its
owned slot/artifact properties. Preserve any additional native launch properties
in the snapshot. The runner must preserve originals, own independent recovery,
verify SSH/dashboard/protected-service health, verify head generation and exact
protected answers, and enforce no worker generation listener or topology change.
Its `ok` follows those real checks. The adapter does not implement or replace the
native guard, create a listener, or claim native acceptance from a file snapshot.

Native runner implementation, accepted neuron artifacts, the queue grant and the
actual Grok Bot restart approval are owner-provided dependencies. A missing one
is an observable blocker. Supply these dependencies to open the native path.

## Verification

```sh
CUDA_VISIBLE_DEVICES='' python -m pytest -q
ruff check .
python -m build
python -m glm_rsi_dream.dream --help
```

The CPU suite trains genuine additions on fictional numerical rows at hidden
dimension 4096 and exercises the CLI lifecycle, signed manual/Automatic authority,
two epochs, corrupt artifacts, stale evaluations, partial rank failure,
interrupted recovery, expired grants and exact rollback after retirement.
CPU synthetic semantic grades are software fixtures, not GLM recall scores.
