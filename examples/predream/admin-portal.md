# RSI Dreambox local portal and API integration

The six portal pages use the real admin API through a same-origin host.
The default hosted portal starts locked. After one password login, `portal.js`
sends `/api/v1` requests with the HttpOnly session cookie. The host loads the
private engine credential and forwards authorized requests to the internal API.
The backend verifies the separate controller-signed grant and optimistic
revision before writing the fixed workspace. An explicit transport preview can
use an in-memory bearer credential; hosted Simple uses the password session.
Head installation and session controls are in [dreambox-head.md](dreambox-head.md).

Dreams and scheduling show **Under development** in Simple and Advanced.
The single server-owned `dream_features_enabled` flag defaults to false.
`/system.features.dreams` publishes the capability; all seven dream and schedule
operations return `503 feature_unavailable: under development` while it is
closed. Portal refresh skips those reads and keeps rules, queue and raw-fact
pockets usable. Stale dream forms and action handlers stay gated. The owner
opens `--enable-dream-features` after dream-neurons passes the pocket-OFF recall
bar. Local tests exercise the explicit opening path with fictional data only.

The OpenAPI 3.1 document is packaged at
`src/glm_rsi_dream/resources/rsi-admin-openapi.json` and available from the portal
at `/openapi.json`. It defines all 38 operations, closed request shapes, typed
responses, revision formats and HTTP errors. The portal builds its operation
forms from `components.schemas`. Simple and Advanced use this same contract.
The API explorer downloads it. Detailed backend behavior is in [admin-api.md](admin-api.md).

| Portal page | API resources | Review boundary |
| --- | --- | --- |
| Rules | `/rules`, `/rules/presets`, `/rules/check`, `/policy` | Literal rule matching and a next-workspace policy profile |
| Learning queue | `/learn`, `/queue`, `/queue/{id}` and grade/approve/reject actions | Prepared intake, pending proposals, holds, versioned edits and signed decisions |
| Dreams and schedule | `/schedule`, `/dreams` and job checks | Under development behind the server flag; qualified opening exposes manual preparation |
| Knowledge pockets | `/pockets` and import/export/extract/copy/merge/rollover | Portable single-topic raw facts with hashes and provenance |
| Autonomy | `/control` | Host-owned mode, signed grant, revision and existing handoff boundary |
| System | `/system`, `/openapi.json` | Capability readback, API contract, credential connection and request log |

Collections return `{"items":[...]}`. Queue revisions are snapshot SHA256
strings; other editable resources use integer revisions. The client sends the
current `expected_revision`. A 409 preserves the form draft for refresh and
review. A hosted 401 returns to Sign in; a 403 displays the denied grant or
Origin. The host engine credential and signed grant are independent. Browser
requests carry the session cookie, and the request log contains methods, paths
and status codes.

Blank optional fact selection omits `fact_ids`, so copy includes all facts.
An explicit empty selection is invalid. Shared pocket import/export preserves
raw JSON text: lexical values such as `1.0` must survive roundtrip for the
Python content hash. The browser keeps the original package text for import.

## Runnable CPU fixture demo

From an installed package or the repository development environment:

```sh
python -m glm_rsi_dream.predream.admin_demo \
  --root /absolute/fresh/fictional-demo --mode harness --port 8767
```

Use a fresh root each time. `--mode spark` exercises the same local library
with Spark-side control settings; it does not contact a Spark. `--port 0` selects
an available loopback portal port. The console prints the portal URL. Open it
and set one temporary portal password. The host loads its private engine
credential; browser actions use the session. The live API demo uses the regular
portal URL. The
separate `?demo=1` mode remains the portal's browser-only fictional mock.

This demo imports a prepared fictional pack through manual intake, scripts eight
explicit safety/conflict and grading protocol responses, leaves three proposals
pending, and signs/stages a fourth. The fixture receipt declares synthetic
judgments and zero model calls. It creates a companion preset and an empty
Aster pocket. Its default server gate is closed and it seeds no schedule or
dream job. It starts the real API and portal.

1. In Learning queue, approve the cobalt hatch, reject the silver dock with an
   owner reason, and review the personal meeting's memory destination. The veto
   preserves the original scripted accept grade alongside the admin rejection.
2. In Advanced, open Knowledge pockets, open Aster, extract the accepted hatch, export its
   shareable package, and copy it. The capacity is one, so rollover opens an
   empty successor while sealing the original.
3. Open Dreams and schedule in Simple and Advanced. Each shows
   **Under development**, with no runnable dream or schedule controls. System
   readback publishes the same server-owned status. Direct API calls remain
   gated even if a client tries to enable the feature in a request body.
4. Browse Rules, Autonomy and System in both layers. Select Manual, Supervised
   or Automatic. The choice applies to upcoming decisions and preserves the
   completed staged fixture. If access has expired, Approve renews the existing
   session scope and applies the choice.
5. Stop with Ctrl+C. Both local listeners close and temporary authority files
   are removed. The fixture pack, queue, staged manifest and receipts remain
   under the selected root as evidence.

The tap stays off. This demo executes CPU fixture protocol and local HTTP only.
It installs no service, calls no model, performs no training and changes no
serving runtime. Its temporary grant authorizes this one fictional root and
expires with process cleanup. Those credentials do not authorize production.

## Contract verification

Install the repository test extras and provide Node 18 or later for the shipped
JavaScript client tests:

```sh
python -m pytest -q tests/predream/test_admin_contract.py \
  tests/predream/test_admin_api.py tests/predream/test_admin_dream_gate.py \
  tests/predream/test_admin_portal.py tests/predream/test_admin_portal_gate.py
```

The contract suite executes `api()` and `parseForm()` from the actual packaged
`portal.js` through the actual preview and backend in both local modes.
Independent JSON Schema validation checks requests and raw responses against
the packaged OpenAPI document. It covers typed nullable/array/numeric/checkbox
forms, read collections, signed approval and veto, holds, stale revisions,
authorization failures, the default dream gate, the explicit host opening path,
manual readiness under isolated fixture configuration and pocket sharing. The Node harness
suppresses DOM startup and supplies form inputs. Visible navigation and form
interaction have separate browser acceptance receipts.
