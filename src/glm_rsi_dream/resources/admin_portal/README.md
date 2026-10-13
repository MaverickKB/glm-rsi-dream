# RSI Dreambox

Simple opens six everyday task pages: Rules, Queue, Dreams and schedule,
Pockets, Autonomy and System. Advanced opens the full policy, detailed editors,
raw values, bulk review, import/export, request logs and the OpenAPI explorer.
The User manual and page help load the shared `help.json` content.

The portal is bundled static HTML, CSS and JavaScript with system fonts and no
frontend runtime dependencies. It requests data on connection, refresh and
actions. It creates no background polling, model requests or collection tap.

The approved artwork uses a matte Zz synapse. `favicon.png` is the
single swappable flat icon shared by the sidebar and browser favicon, copied
unchanged from `LOOP2-shared/brand/icon-zz-64.png`. `hero.png` is the unchanged
`hero-zz-256.png` used by first-run and login. Both artwork files load before
sign-in so the locked entry screens can display them.

## Default head hosting

The packaged install targets the Spark head, beside the engine, in its own
small process and port. It loads a private local engine credential. First-run
sets one portal password. Sign in once; normal navigation and changes reuse
an expiring HttpOnly session cookie. The engine credential stays on the host.
Installation and the systemd unit are prepared locally; deployment requires
the owner's go. See the head-host runbook supplied with the host package.

System contains Security settings. Simple offers session timeouts of 15 min,
1 hour, 8 hours and 1 day. Advanced offers custom values from 5 min to 1 day.
The single server value is `session_seconds`; changes apply to new sign-ins.
Activity never extends the original expiry. Sign out other sessions preserves
this browser's session. Password changes invalidate older sessions. Advanced
alone can turn the portal lock off after a short warning.

Advanced also exposes Connect to an engine with its URL and API key. The host saves that explicit target and key in a private local file, proxies
requests and keeps the browser on its own origin. The default hosted flow uses
the host-loaded credential and has no browser key field.

## Temporary local preview

From an installed package, explore fictional browser-only data:

```sh
python -m glm_rsi_dream.predream.admin_portal --port 8767
```

Open `http://127.0.0.1:8767/?demo=1`. Demo edits last for that page session.
Demo access is explicitly selected and API errors remain visible.

For a prepared workspace, use the existing admin API and host-issued signed
permission described in `examples/predream/admin-api.md`. Create a separate
private API bearer file, then start the API and preview:

```sh
python -m glm_rsi_dream.predream.admin_api init-token /private/admin-api.token
python -m glm_rsi_dream.predream.admin_api serve \
  --root /absolute/prepared/workspace --token-file /private/admin-api.token \
  --port 8768 --allow-origin http://127.0.0.1:8767
```

In another terminal:

```sh
python -m glm_rsi_dream.predream.admin_portal \
  --port 8767 --api http://127.0.0.1:8768
```

Open `http://127.0.0.1:8767/`, switch to Advanced, and select Connect to an
engine. Use that preview URL and the API key. The key stays in browser memory. The API checks the fixed workspace, optimistic
revision and signed authority for every change. The preview passes authorization
and origin through to the API. It binds loopback and accepts a loopback API
origin. These commands create temporary local processes.

Both servers default to a 64 MiB request limit. An operator can set
`--max-body-bytes` on both for an approved larger import.

## Everyday actions

Queue presents the model proposal and its source evidence. Approve applies that
proposal, including a model rejection. Reject records an explicit admin veto and
reason. Edit or hold opens review controls; changed text or evidence creates a
fresh candidate through the API. Existing sealed handoffs retain their policy.

Dreams, scheduling and consolidation show **Under development** in Simple and
Advanced. A single server flag defaults off; all dream and schedule API reads
and actions return `feature_unavailable: under development`. The portal reads
`system.features.dreams`, skips unavailable requests and provides no browser
switch that opens the gate. Browser-only demo also keeps this gate closed.

The dream-neurons owner must pass the pocket-OFF recall bar before an operator
opens the server flag with `--enable-dream-features`. That explicit host path
permits manual plan preparation through the existing bearer and signed grant;
it performs no native training. Missing feature metadata keeps the UI closed.
Pockets holds raw facts and provenance; copy, share, import, merge and rollover
manage those catalogs while the dream gate is closed.

Advanced policy edits save a validated profile for a new workspace. Export its
JSON for the existing `--policy-config` entrypoint. Active sealed policy remains
visible in raw values. Autonomy changes require the host's existing signed grant.

Keyboard controls: Tab follows native focus order, Escape closes a flyout,
`A` switches Simple/Advanced outside editors, `/` focuses Queue search, and
Ctrl+Enter or Command+Enter opens confirmation for selected queue proposals.
Bulk actions record each item separately; after an error, completed items are
removed from the selection and remaining items stay available for review.

Traffic collection stays off and undeployed during development. System shows
the required per-user exclusions, approved window and explicit owner go. These
are design requirements for the later collection surface.
