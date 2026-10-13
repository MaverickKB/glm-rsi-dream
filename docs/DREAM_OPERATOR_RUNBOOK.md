# RSI Dreambox development operator runbook

Under development. New dream work is closed by default through the host-owned `dream_features_enabled` flag. The dream-neurons owner must first pass the pocket-OFF recall bar. The host can then open the same development flag with `--enable-dream-features` in the API or CLI. Portal development access prepares manual plans. It is separate from native learning acceptance. Existing signed rollback and interrupted-operation recovery remain available to restore an already-authorized epoch.


This runbook describes the gated artifact and lifecycle contracts for a future consolidation path. CPU fixtures exercise preparation, paired state changes, retirement and recovery; native factual consolidation remains unverified. The trigger supports GLM-5.3-Flash, a frozen base and a TP=2 pair. Its current fitter runs on CPU.

The portal's dream job returns a manual preparation plan. Its raw-fact pocket catalog supplies knowledge and provenance. The trigger starts from accepted attention artifacts and sealed training packets. Use the [task manual](admin/README.md) for portal planning and [lifecycle contract](DREAM_LIFECYCLE.md) for input schemas and the native runner protocol.

## Prepare the handoff

Choose the existing private LOOP2 root. Preserve its host signing key, candidate receipts, accepted teacher files and evaluation files. Use a unique run ID. All capitalized paths and IDs below are operator-supplied placeholders.

The host admin provisions `PREDREAM_ADMIN_KEY_FILE`, a private key owned by the operating user, and `PREDREAM_ADMIN_PERMISSION_FILE`, a signed grant for that root, execution mode and autonomy. The grant needs `dream_register`, `dream_start`, `dream_qualify`, `dream_activate` and `dream_retire`; accepted candidate verification also needs `stage`. Imported learned catalogs need `catalog`. The trigger branch adds explicit `predream admin-grant --action` values. Preserve existing permissions when provisioning the grant.

Manual and supervised lifecycle commands require `--confirm`. The Automatic level may omit it only with an explicit grant for `automatic` and that action. The source and evaluation checks apply at every level.

For a native activation, the owner must provide a pinned runner and `DRIVER.json`. Verify an exact grant line beginning `GRANT DREAM-trigger ` in the neurons owner's GPU queue and the real owner-authorized paired-reload approval. The approval receipt binds the allowed epochs and expiry. Keep both ranks together, generation on the head, and the worker generation listener absent. The native owner captures the rollback target and protects SSH, dashboard and adjacent services. A receipt file records actual authority.

For a CPU rehearsal, use a fictional filesystem pair with `--harness-pair PAIR.json` and qualify with `--evidence-kind harness`. Harness evidence stays in that rehearsal. Spark activation requires native evidence and the native driver.

## 1. Inspect current state

```sh
glm-dream-trigger --root ROOT status
```

Read the active run and `operation_status`. If an operation is pending, follow Recover below before starting new work. Save the current state with the run evidence. Keep the host key and pinned runner accessible throughout the lifecycle.

## 2. Register accepted teachers

```sh
glm-dream-trigger --enable-dream-features --root ROOT register-pocket POCKET.json --confirm
```

Repeat for each source pocket. The spec names candidate IDs, both rank artifact paths and hashes, acceptance report, recall group and minimum recall. Check that the report names these exact teacher tensors. The attention owner supplies genuine acceptance evidence.

An ordinary source candidate must still be accepted or staged, routed to that pocket, and bound to current valid approval. A previously accepted teacher can use the operator's imported learned catalog, exact statements and receipt. Registration verifies its catalog digest and acceptance mapping while preserving `grade:null`.

Success: the pocket is registered with status `accepted` under `ROOT/dream/pockets/`. Preserve its manifest digest for the training plan. Registering identical sealed input is idempotent; changed input needs a new identity and review.

## 3. Fit a new version

Have the capture owner assemble paired training packets with training rows, frozen-base digest and source-pocket manifest digests. Verify the plan's ordered ranks are 0 and 1.

```sh
glm-dream-trigger --enable-dream-features --root ROOT start --run-id RUN --pocket POCKET --training TRAINING.json --width 32 --steps 1000 --confirm
```

Repeat `--pocket` for multiple teachers. Width is added neurons per rank, default 32; total width is twice that value. Steps is fitting iterations, default 1000. Both must be positive. These choices are training settings to qualify, not recall guarantees.

Success: run status `trained`, with both tensor files and a sealed manifest under `ROOT/dream/artifacts/RUN/`. The addition uses gate/up/down tensors in the existing always-on shared expert. Epochs are reserved monotonically. Failed training keeps its reserved epoch and partial files; inspect them and start a new run ID after correcting the input or fitter failure.

## 4. Qualify with every teacher OFF

The native evaluation owner loads this exact candidate under its guarded trial, disables every attention pocket and clears conditioned caches. It supplies the sealed suite, complete baseline, report, raw generated replies, semantic judgments and both rank readbacks. Training rows and a fresh held-out acceptance set serve separate purposes.

```sh
glm-dream-trigger --enable-dream-features --root ROOT qualify --run-id RUN --suite SUITE.json --baseline BASELINE.json --report REPORT.json --evidence-kind native --confirm
```

Qualification replays the owner's evidence and seals it. It checks artifact epoch, base and both neuron hashes, pocket-OFF state, empty active memory and clean caches. Ordinary and protected replies must match their baseline exactly. Leakage and recall use the recorded semantic judgments. Every source meets its registered minimum recall with its facts retained in the rubric.

Native qualification also requires at least 60 ordinary, 8 protected, 12 leakage and 24 precision rows. A later epoch retains every previous knowledge and control case. The trainer fits a replacement addition, so cumulative recall must survive each replacement.

Success: status `qualified`, with an immutable manifest and host-signed qualification. A failed check preserves evidence for inspection. Correct the actual evaluation or candidate failure and qualify a genuinely passing artifact before activation.

## 5. Activate through the paired guard

```sh
glm-dream-trigger --enable-dream-features --root ROOT --native-driver DRIVER.json activate --run-id RUN --confirm
```

The driver prepares both ranks. The trigger requires the artifact's parent to match the current active epoch and verifies that source slot hashes match its teachers. It saves the exact previous pair, catalog and signed rollback receipt before mutation. A compare-and-swap precondition rejects a runtime changed by another owner.

Success: status `active`, exact target readback on both ranks, every attention pocket OFF and caches clean. Teacher files and slots remain preserved at this step. Independently check the native owner's receipt for head generation, worker listener absence, protected replies and protected service health. The generic driver adapter delegates those real checks to the pinned owner runner.

A failed reload or readback attempts immediate exact recovery. Read the failure and journal status. Use Recover if the operation remains pending.

## 6. Retire the consolidated source slots

```sh
glm-dream-trigger --enable-dream-features --root ROOT --native-driver DRIVER.json retire --run-id RUN --confirm
```

Retirement requires this run to be the active qualified artifact, the same epoch/base/rank hashes, matching teacher slots, all pockets OFF and clean caches. It frees only that artifact's source slots. Preserve source files for provenance and rollback.

Success: status `retired`; source catalog entries are retired; affected LOOP2 candidates have `learned` receipts bound to the qualified artifact. Inspect the owner's paired readback and the latest candidate receipts. A portal catalog deletion is a separate raw-fact storage operation.

## Roll back the active epoch

```sh
glm-dream-trigger --root ROOT --native-driver DRIVER.json rollback --confirm
glm-dream-trigger --root ROOT status
```

Rollback uses the signed receipt saved at activation. It restores the exact preceding pair, pocket slots, catalog and affected candidate snapshots, including slots retired after activation. Newer unrelated candidate decisions remain intact; append-only queue history remains available. A second epoch rolls back to its preceding epoch.

Success: the readback equals the saved pre-activation pair and the source candidates have their previous latest state. Have the native owner repeat its original head-generation, protected-answer and control-plane probes. Keep the source files, host key and pinned runner until recovery has been verified.

Saved activation authority opens this exact recovery path after a grant expires. It authorizes restoration of the saved target. Manual and supervised rollback still need confirmation.

## Recover an interrupted operation

```sh
glm-dream-trigger --root ROOT --native-driver DRIVER.json recover
glm-dream-trigger --root ROOT status
```

Recover verifies the signed pending journal and restores its saved pair, candidate states and lifecycle state. It needs the preserved host key and owner runner. It requires no new grant or confirmation, keeping the bounded repair lane available after expiry.

Success: the operation journal says `recovered` and the exact saved pair is read back. If recovery fails, retain the pending journal and raw failure evidence. Restore the required key, runner or native guard dependency through its owner, then recover the same operation. New lifecycle work waits for that restoration.

## Record the outcome

Keep the command output, artifact and evaluation hashes, paired before/after readbacks, native guard receipts and queue changes beside the private run. Record CPU rehearsal, native qualification, activation, retirement and rollback as separate observed states. This documentation task adds no native acceptance claim.
