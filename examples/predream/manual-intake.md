# Prepared knowledge in RSI Dreambox

Prepared knowledge uses the normal queue, GLM grader, routes and staging.
Dream features remain Under development behind the default-off host flag.

## Portal and API

In Simple, open Queue, choose **Learn this**, paste a note or upload a UTF-8
`.txt`, `.md` or `.json` file, then choose **Add for review**. File content appears
in the editable note before submission. Text uses GLM extraction; a `.json`
upload declares facts directly. Simple paste uses text format and source reference
`manual:portal`. Upload uses `manual:file:FILENAME`.

Advanced adds **Input format** and **Source reference**. Choose JSON to paste a
facts pack directly, or text for GLM extraction. Keep the same reference for an
exact replay; use a stable reference that identifies the prepared source.

The portal queues and grades each pending unreviewed fact in order. It stops on a
failed model review and keeps the queued facts available. Close the error panel
and choose **Grade fact** to retry. Review the source, proposed destination and
model reason before approving or rejecting. In manual autonomy approval is a
separate signed decision. Supervised and Automatic use the same controller and grant.

The API accepts `POST /api/v1/learn` with this closed request shape:

```json
{"source_ref":"manual:lumen-spec","format":"json","content":"{\"facts\":[{\"statement\":\"Fictional rover Lumen has three sample bays.\"}]}"}
```

Use `format: text` for a note. Content must be nonempty and at most 1,000,000
characters. The portal file limit is 1,000,000 bytes and rejects invalid UTF-8.
The source reference is nonempty and at most 512 characters. A JSON pack contains
1 through 2000 facts. Each row has only `statement` and optional `prompt_evidence`,
both nonempty strings of at most 16,000 characters. Duplicate statements in one
prepared pack are rejected. Credentials, including decoded JSON values, are
rejected at the API before persistence or model access.

Intake returns current `items`, newly `queued_ids`, `novelty_check: skipped_manual`
and `secret_preflight: clear`. Send an item's current revision to
`POST /api/v1/queue/{id}/grade` as `{"expected_revision":"SHA256"}`. The response
has `item` and `errors`. A stale or concurrently edited fact returns 409 and keeps
the edit. A failed GLM review reports a retryable error with the fact pending.
Existing proposals and decided facts replay without model calls. Exact intake
replay reuses its extraction receipt and queue IDs; changed content creates a new
version and preserves old facts. Both calls require bearer access and a scoped
signed `approve` permission. Host GLM settings and authority cannot be supplied in
the request. See [API setup](admin-api.md).

For a temporary fictional CPU proof, run from the source checkout:

```sh
PYTHONPATH=src python examples/predream/manual_cpu_demo.py
```

The script prints its loopback portal URL and the path to a private temporary
bearer token. Read that token locally into Admin access. It accepts only the three
printed fictional facts and deliberately fails the first three Orin review
attempts to exercise retry. Its responses are fixed CPU fixtures. Ctrl-C closes
both servers and removes the workspace and credentials. It contacts no model.

## CLI

On the head Spark, set ROOT to a private directory under
`/srv/dreambox/workspace/`. Run the installed package:

```sh
python -m glm_rsi_dream.predream --root "$ROOT" learn prepared.txt
python -m glm_rsi_dream.predream --root "$ROOT" queue list
python -m glm_rsi_dream.predream --root "$ROOT" grade --all-pending
python -m glm_rsi_dream.predream --root "$ROOT" queue list
python -m glm_rsi_dream.predream --root "$ROOT" approve --id CANDIDATE_ID
python -m glm_rsi_dream.predream --root "$ROOT" absorb
python -m glm_rsi_dream.predream --root "$ROOT" stage --run-id prepared-01
```

Deployment mode and approval use the shared admin control. The default autonomy
is manual: GLM grades remain pending proposals until an authorized admin approves
each ID. Set up the shared host-admin key and a bounded permission grant through
`admin-init` and `admin-grant`; keep those commands outside model tools. The host
admin supplies `PREDREAM_ADMIN_KEY_FILE`, `PREDREAM_ADMIN_PERMISSION_FILE`, and the
`spark` or `harness` deployment mode. `control-status` reports current authority.
Supervised and Automatic settings require their own explicit grants. Source=manual
means prepared knowledge; it is independent of the autonomy setting. Queue listing
shows `propose:accept` or `propose:reject`, proposed destinations and named pockets.
Absorption and staging require the shared signed decision and handoff authority.

Text files use GLM extraction. JSON packs declare atomic facts directly:

```json
{"facts":[{"statement":"Paris is the capital of France."},
          {"statement":"Fictional rover Lumen has three sample bays."}]}
```

Every candidate has `source.kind=manual`, `novelty=unmeasured`, an explicit
`intake.novelty_check=skipped_manual` receipt, and zero blind-probe time.
The user deliberately requests retention, so known facts and equivalent learned
facts remain eligible. Grounding, atomicity, safety and routing still apply.
Credential-shaped input is redacted before model access and queue persistence.
GLM reviews safety and conflicts, then makes the accept/reject and route decision.
An unsafe review cannot be overridden by a custom grading policy. A failed review
or inconsistent grade leaves the item pending and reports an error.

Existing learned facts require an explicit operator-supplied catalog:

```json
{"facts":[{"id":"lumen-bays","statement":"Fictional rover Lumen has two sample bays.",
           "receipt":"fixture:previous-learning-KEEP-receipt"}]}
```

```sh
python -m glm_rsi_dream.predream --root "$ROOT" queue import-learned learned.json
python -m glm_rsi_dream.predream --root "$ROOT" learn prepared.json
```

`learn prepared.json --learned-facts learned.json` also registers that catalog.
Catalog receipts are operator attestations of prior learning. Import performs no
training and does not independently verify a native learning result. Staged or
accepted candidates never become learned merely through import inference.
The external catalog id is retained; the queue id follows the normal source/text
hash contract. Keep each catalog immutable. Changed learned content must enter
through intake as a proposed update.

Before grading manual candidates, or ordinary candidates when learned receipts
exist, GLM compares each claim with the learned catalog. A contradiction becomes
`change.kind=update` with `change.targets` containing old queue IDs, statements,
content hashes and provenance receipts. An equivalent claim is a duplicate.
Queue listing shows source, relationship and target IDs. An update overrides
known/duplicate novelty rejection. Manual equivalent claims also bypass that drop.

The original remains `learned` while the proposed replacement becomes accepted
and, after admin approval, staged. The raw staging manifest seals update metadata and rechecks the old
fact and receipt before reuse. Memory-routed candidates use normal absorption.
Weights and pocket candidates stage as raw prepared knowledge. The existing
human-triggered dream remains Under development until its host gate is opened
after the owner passes pocket-OFF recall.
Staging is a preparation handoff; it does not apply replacements to model weights.

Text extraction receipts are persisted before queue appends. Repeating the same
file/pack and source reference reuses extraction and avoids duplicate queue rows.
Queue source references include manual intent and the file content hash. A file
previously rejected by automatic novelty filtering can re-enter as deliberate
manual input. Changed evidence creates a fresh candidate even when the proposed
statement stays the same. Earlier sealed manual receipts keep their original IDs
on replay. Changing the file creates a separate extraction receipt. Use a new staging run
id for newly accepted facts. Funnel timings include `manual_intake` and
`safety_conflict`, with normal grading and handoff timings. No dream timing is
needed. Policy configuration for the grade step remains supported.

RSI Dreambox accepts prepared text or JSON through `POST /api/v1/learn`, with
`source_ref`, `format` and string `content`. Uploaded knowledge follows the same
manual queue as the file CLI. Grading uses `POST /api/v1/queue/{id}/grade` with
its current revision. The portal grades serially, keeps manual proposals for
review, and offers a retry when a fact remains pending after a grading error.
Secrets are checked before storage or model access. These actions stage no
training. The head-hosted portal loads its engine credential and uses one
password session; remote engine settings belong in Advanced.
