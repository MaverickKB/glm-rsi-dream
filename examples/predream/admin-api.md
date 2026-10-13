# RSI Dreambox API

This local backend manages LOOP2 save rules, prepared knowledge, queue review
and portable raw-fact pockets. It runs the same library in `spark` and `harness`
modes. The fixed workspace root is selected at startup. No request accepts a
filesystem root or executable command.

Dreams and scheduling show **Under development** and are closed by default.
One server-owned startup flag, `dream_features_enabled`, gates all seven dream
and schedule operations, including reads. Closed endpoints return HTTP 503 with
`{"error":"feature_unavailable: under development"}` before body validation or
storage access. `/system.features.dreams` publishes the flag, status, label,
owner and opening criterion. Requests cannot change it. The dream-neurons owner
opens `--enable-dream-features` after the pocket-OFF recall bar passes. The local
test fixture checks the explicit opening path; it establishes no native recall
qualification. Queue, rules, grading, policy and raw-fact pockets remain usable.

The default request limit is 64 MiB. The host can set `--max-body-bytes` for
larger approved imports. Pocket capacities are 1 through 2000 raw facts; rollover
opens a successor catalog. The limit is explicit in the OpenAPI request shapes.

The packaged contract is `src/glm_rsi_dream/resources/rsi-admin-openapi.json`.
Authenticated routes start at `/api/v1`; `/openapi.json` serves the contract.
Request bodies are JSON. Errors return `{"error":"..."}`. Stale revisions return
409. Rule, pocket, dream and schedule revisions are integers; queue revisions are
full snapshot SHA256 strings. Collection responses use `{"items":[...]}`.

The host configures `PREDREAM_ADMIN_KEY_FILE`, `PREDREAM_ADMIN_PERMISSION_FILE`
and `PREDREAM_MODE` using the existing LOOP2 admin CLI. The HTTP bearer token is
a separate private file, created by `python -m glm_rsi_dream.predream.admin_api
init-token FILE`. It never grants learning authority by itself. Mutations require
the existing signed grant for the selected root, mode and autonomy.

Start a temporary local instance with:

```sh
python -m glm_rsi_dream.predream.admin_api serve \
  --root /absolute/prepared/workspace --token-file /private/admin-api.token \
  --port 8789 --endpoint http://127.0.0.1:8888 \
  --model GLM-5.3-Flash-EXL3 --timeout 120
```

The server binds loopback. A portal may use a same-origin proxy, or an explicit
`--allow-origin http://127.0.0.1:5173` development origin. Authorization uses
`Authorization: Bearer TOKEN`; tokens are never URL parameters. This lane builds
and tests the backend locally. It does not install a service on the head.

The host selects the GLM endpoint, model and timeout at startup. Request bodies
cannot replace them. The preview's `--api-timeout` defaults to 900 seconds so
model protocol repair can complete within the same request. Use the owned model
endpoint only when native access is authorized.

Dream and schedule routes default to HTTP 503 with
`{"error":"feature_unavailable: under development"}`. `/system.features.dream`
reports the current gate, its owner and opening criterion. The single host flag
`--enable-dream-features` opens development preparation after dream-neurons proves
pocket-OFF recall. A request, signed admin grant or browser demo cannot open it.
Queue, rules, grading and raw-fact pockets work while the gate is closed.

Save rules declare literal match patterns, user inclusions/exclusions, source
kinds, priority and domain (`general`, `companion`, `persona`). Their action is
`propose` or `ignore`, with an intended destination. Presets supply editable
companion and persona templates. `/rules/check` evaluates supplied synthetic
context and returns matches and the highest-priority rule. It performs no model
judgment or traffic capture. GLM and signed controller authority still own
learning decisions. Collection remains OFF and undeployed.

Queue edits use the complete current revision. Hold/note edits append snapshots
and invalidate applied decisions. Editing knowledge creates a new manual
candidate, clears the old decision and holds the previous candidate. The new
candidate needs the ordinary GLM safety/conflict/grade path and explicit approval
in manual autonomy. Staged, absorbed and learned records are immutable here.
`/queue/{id}/approve` delegates the existing controller's GLM-proposal approval.
`/queue/{id}/reject` applies a signed admin veto with an explicit reason, preserving
the original GLM verdict. A subsequent queue edit reopens that item for review.

`POST /learn` accepts `{source_ref, format, content}`. `format` is `text` or `json`.
Text uses GLM to extract atomic facts; JSON declares them in
`{facts:[{statement,prompt_evidence?}]}`. Both skip the blind novelty probe and
still require ordinary GLM safety, conflict and destination review. Credentials
are rejected before model access or persistence. See [prepared knowledge](manual-intake.md)
for bounds, replay behavior and the portal paths.

The intake response has `items`, `queued_ids`, `novelty_check: skipped_manual`
and `secret_preflight: clear`. `POST /queue/{id}/grade` takes the item's current
`expected_revision` and returns `{item, errors:[{id,error}]}`. Both operations
require bearer access plus the scoped signed `approve` permission. The portal
grades pending unreviewed items serially, stops at a failed review and offers
`Grade fact` to retry. In manual autonomy the resulting proposal remains pending
until the separate approve or reject action. Replaying a reviewed or decided item
does not call the model again. Neither operation trains or activates a dream.

`GET /policy` exposes all 17 current extract/grade/control knobs using their owning
component defaults, plus any immutable active policy. `PUT /policy` validates and
saves an editable next-workspace profile. Export its `policy` object as JSON and
use it with `--policy-config` in a new workspace. Existing experiment policy and
handoff receipts stay sealed. Selecting Automatic in a profile grants no authority;
the host's signed permission must authorize it when that profile is used.

When the development gate is open, schedules use IANA timezones, weekday numbers
(Monday=0), approved windows and
idle/active-request thresholds. Overnight windows belong to their start day.
An authenticated admin registers a staged run as a dream job. `/dreams/{id}/check`
evaluates an explicitly supplied timestamp and idle context, verifies current
authority and sealed staging, then returns eligibility and a manual preparation
plan. This is a scheduling handoff, with the human confirmation path described
by LOOP2's existing dream CLI. The API executes no training or serving mutation.

Pockets are single declared-topic raw-fact catalogs with capacity and provenance.
Extract copies accepted/staged reusable queue items with valid approval receipts.
Export adds a content hash; import checks it, content, capacity and credential
patterns. Copy may select fact IDs. Merge requires the same declared topic,
preserves provenance, rejects conflicting duplicate IDs and creates a new pocket.
Rollover seals a full pocket and opens an empty successor. Deletion removes active
membership while durable history preserves the removed state. These operations
prepare knowledge for learning; they do not compose or activate tensor artifacts.

Autonomy levels are `manual`, `supervised` and `automatic`, shown as Manual,
Supervised and Automatic. `PUT /control` immediately applies a choice within the
host session. Completed staged, absorbed and learned rows keep their receipts;
unfinished decisions reopen for review. `POST /control/approve` accepts the same
Control body and returns ControlView. The explicit admin action can renew an
expired, correctly signed session for 16 hours within its existing root, modes,
actions and allowed levels. The prior grant bytes are archived under
`admin/permission-history`. Invalid signatures, scope, modes, actions and
unsupported levels remain denied. New grants, policies and CLI choices use
`automatic`; earlier versioned experiment receipts retain their original bytes.

Prepared knowledge uses `POST /learn` with the closed body
`{source_ref, format: "text" | "json", content}`. JSON content is a facts pack;
text content is extracted by the host-selected GLM client. Both use manual
intake, skip blind novelty filtering, reject credentials before persistence and
retain ordinary safety, conflict and routing checks. The result includes typed
queue items and newly queued IDs. `POST /queue/{id}/grade` takes its current
`expected_revision`; it returns `{item, errors}`. Manual autonomy keeps proposals
pending until separate approval or rejection. Reviewed items replay without
model calls. Concurrent edits and holds preserve their current queue revision.
The host selects endpoint, model and timeouts; request fields never select them.
There are 38 typed admin operations, plus the head portal session routes in the
head-hosting guide.
