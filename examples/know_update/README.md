# Learned-fact update evaluation

This suite tests correction and replacement of facts learned by GLM-5.3-Flash.
It has three fictional domains, 15 facts per revision, three linked revisions,
18 bare recall questions, 12 ordinary controls, four protected controls and four
leakage questions. Source facts and evaluation answers stay outside generation
requests. No claim applies to another model.

The pocket trajectory measures baseline, original acquisition, correction and
replacement. The dream trajectory measures baseline, original consolidation,
correction staged over the original neurons, corrected consolidation,
replacement staged over the corrected neurons and replaced consolidation.
Consolidated phases require every attention pocket and its active rows OFF on
both ranks. Staged phases explicitly retain the older neuronal catalog and
require the new pocket catalog to override it. This tests updating already
consolidated knowledge as well as moving the latest pocket into neurons.

A changed fact has a fresh ID and a link to its exact predecessor. Changed
endpoints and calculated capacities are updated dependents. Stable policies
retain their fact IDs and depend on revised ancestors. The third revision's
judge sees every retired version. Latest-value questions and joint rewordings
probe stale recall without putting an old or new answer in the question.

Fixture questions are held out from training and checkpoint selection. They
are fixed across revisions, so these are repeated revision-transfer probes,
not independent novel-fact or broad linguistic-generalization results. After a
native run exposes them, preserve that run and use another frozen suite for any
new training or selection informed by its answers.

## CPU verification

Use Python 3.11 or 3.12 from this checkout. The CPU runner imports only the
standard library. The existing development environment provides pytest and
Ruff; the narrow local `.venv` has pytest 8.4.2.

```sh
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m pytest -q tests/know_update
ruff check src/glm_rsi_dream/know_update tests/know_update
PYTHONPATH=src .venv/bin/python -m glm_rsi_dream.know_update check
PYTHONPATH=src .venv/bin/python -m glm_rsi_dream.know_update simulate --arm pocket
PYTHONPATH=src .venv/bin/python -m glm_rsi_dream.know_update simulate --arm dream
```

`simulate` is a scripted fixture oracle. It performs no model inference, network
connection, process launch, training or serving operation. Its passing reports
have `evidence=cpu-simulation` and `native_acceptance=false`. `--output` records a
suite copy, flushed raw event journal and sealed report in an unused directory.
Without `--output`, CPU commands create no run directory. The report seal checks
content integrity; it does not establish authority or native provenance.

## Native opening criteria

The DREAM-neurons owner grants the GPUs in its GPU-QUEUE.md. This lane requires
an owner-supplied native adapter and a structured grant binding its source hash.
No such adapter or grant has been supplied to this lane. The inspected legacy
`gradient_pocket.manage_attention --replace` command restarts serving, so its use
also requires owner approval. The evaluator does not run that
command or synthesize that approval.

An owner grant is a single line using this schema, with real values:

```text
GRANT KNOW-update-pocket-01 {"job":"KNOW-update","arm":"pocket","expires":"ISO8601 with timezone","generation":true,"transitions":true,"adapter_sha256":"SHA256 of approved adapter","ken_restart_approval_ref":"actual Grok Bot approval reference if restarts are needed"}
```

For the dream arm request a separate grant naming that arm. `REVOKE request-id`
closes an earlier grant. Expiry, revocation and adapter bytes are checked before
each transition, generation and model judgment. The owner owns grant issuance;
operators do not write their own grants. Recovery remains available after grant
expiry, with generation restricted to the original exact control prompts.

Run from this checkout with the actual owner adapter path and an unused private
output directory:

```sh
PYTHONPATH=src .venv/bin/python -m glm_rsi_dream.know_update native \
  --arm pocket --adapter /absolute/path/to/owner_adapter.py \
  --queue ../../DREAM-neurons/GPU-QUEUE.md \
  --output /absolute/path/to/unused/private-run
```

Grant admission and frozen-suite verification happen before adapter import or
run-directory creation. The adapter module must have no import-time effects.
It declares boolean `REQUIRES_SERVING_RESTART` and exports
`open_backend(training, grant)`. The factory must be side-effect-free too;
resource mutation begins in `transition`. `training` contains model scope,
generation settings and revision facts only, with evaluation questions omitted.
The backend owns cancellation, paired safety, bounded resource use, existing
service protection, owner-scoped candidate tracking and restoration. It owns
no worker generation API and never splits the TP=2 pair.

## Backend interface

The runner uses these methods:

- `inspect_runtime(probe_id)`: actual two-rank readback using the schema below.
- `transition(spec)`: original acquisition, linked correction/replacement,
  consolidation or re-consolidation as named by `spec.action`. The payload has
  current facts, retired IDs and previous/current catalog hashes. Return a
  receipt binding `previous_facts_sha256` and `facts_sha256`. A failed learning
  or installation result must raise or expose unchanged actual runtime state.
- `generate(request)`: exact bare OpenAI-shaped question request sent only to
  the head API. The request has deterministic parameters, one user message and
  an opaque `metadata.rsi_probe`. Return the actual full reply and token usage.
- `judge(case, answer)`: a separate facts-aware judgment on the head GLM. Use
  `judgment.request_for` and `judgment.parse` if helpful. Return boolean
  `correct`, `grounded`, `stale` and a reason for recall, or boolean `leaked`
  and a reason for leakage. Facts and expected values enter judgment only.
- `rollback()`: restore the captured pre-test service, clear test caches and
  retire every owned candidate/trainer. Return explicit
  `owned_candidates_remaining: []` and `owned_trainers_remaining: []` from
  actual inventory. The runner then proves original carrier/runtime identities
  and every original exact control reply again.

The adapter supplies runtime truth; model assertions or a requested config flag
are insufficient. Catalog hashes use `runner.digest` on the ordered full fact
list. Artifact hashes are actual loaded rank files. `pair_sha256` identifies
host/rank/topology configuration, `base_sha256` identifies the frozen base, and
`runtime_identity_sha256` binds actual serving container/image/config/source.
Before testing, the evaluator captures the real original service catalog,
artifacts and control replies. Initial carrier state may contain existing
pockets. The test baseline disables carriers, while rollback restores the
original service. Newer serving epochs may differ after lifecycle changes.

```json
{
  "probe_id": "current opaque probe ID",
  "epoch": "actual serving/load epoch",
  "model": "GLM-5.3-Flash",
  "generation_api_rank": 0,
  "pair_sha256": "verified 64-character hash",
  "base_sha256": "verified 64-character hash",
  "facts_sha256": "effective current fact catalog hash",
  "ranks": [
    {
      "tp_rank": 0,
      "tp_world": 2,
      "generation_listener": true,
      "runtime_identity_sha256": "verified 64-character hash",
      "stale_cache_cleared": true,
      "pocket_enabled": true,
      "pocket_facts_sha256": "current pocket catalog hash or null when OFF",
      "pocket_sha256": "actual loaded pocket artifact hash or null",
      "enabled_pockets": ["actual enabled pocket names"],
      "active_memory_rows": 96,
      "neuron_enabled": false,
      "neuron_facts_sha256": null,
      "neuron_sha256": null,
      "architecture": {
        "hidden_size": 4096,
        "base_intermediate_size": 2048,
        "shared_experts": 1,
        "routed_experts": 288,
        "base_frozen": true,
        "shared_expert_additions": 0
      }
    }
  ]
}
```

Include rank1 with `tp_rank=1`, `generation_listener=false`, its own loaded
artifact hashes and runtime identity. Both catalogs must match the requested
phase. When neurons are active, `shared_expert_additions=1`. The adapter reads
actual model geometry and frozen-base identity; it cannot grow routed experts.
When attention pockets are OFF, names are empty and active rows are zero.
`stale_cache_cleared` must cover prefix/KV state conditioned on all previous
revisions, including caches from older neuronal artifacts.

Runtime state is checked before/after each reply and across its phase. Base
and pair bindings must remain constant across revisions. Any runtime/generation
failure stops further candidate GPU calls and attempts recovery; skipped rows
remain in denominators. Factual failures retain later revision measurements.
A checkpoint that fails recall or preservation is never a passing candidate.

Exact controls compare the entire assistant message, UTF-8 content, finish,
actual completion token count and optional token hash. Protected replies must
finish with `stop`; ordinary replies may retain an identical baseline `length`
finish. Tool/refusal/reasoning metadata changes are differences. Factual answers
must finish with `stop` and must be direct answers.

A passing native report requires every active phase, exact controls, leakage
and restored original service to pass. Preserve raw judgments as GLM
self-review evidence. Independent human grounding/staleness review of raw
answers is a separate acceptance item. Portability has its own acceptance path in DREAM-neurons.
