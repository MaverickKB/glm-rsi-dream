# Opt-in traffic tap

This implementation observes completed head-port8888 requests. It stores only
the current user text, then uses the existing GLM extraction and blind novelty
protocol to append pending candidates to `queue/candidates.jsonl`. The serving
application owns the original generation. The tap performs no proxying or
listener changes. Base weights, shared experts and attention pockets are outside
its collection responsibility.

Offline qualification covers both Spark and external harness callbacks using
recorded GLM responses and synthetic fault tripwires. A running head callback,
fresh extraction quality on ordinary traffic and live throughput remain separate
acceptance targets. This branch has no head installation. Deployment or serving
restart needs owner approval, and serving/GPU activity needs the
DREAM-neurons GPU queue grant.

## Consent and opening the gate

The supplied `tap-defaults.json` has `enabled:false`, an empty principal list and
an empty endpoint. It creates no capture store or candidate queue. The opening
criteria are all of: administrator enable, exact head endpoint, an allowlisted
principal, and trusted per-request `opt_in=True`. Neither the user message nor
request metadata can grant consent. The integrator resolves principal identity
and opt-in from its authenticated UI/session state. Keep the policy and consent
controls owned by that integrator, outside untrusted prompt input.

For an approved installation, set `mode` to `spark` or `harness`, `endpoint` to
the head's exact HTTP(S)8888 API base, and `principals` to local opaque identity
labels. Explain to the opting-in user that literal user evidence and GLM-derived
candidates may enter the existing review queue. That queue has its own retention
policy. Per-request consent covers this pending intake, and grants no learning or
approval authority.

Use one private root and policy per deployment mode. Mode and endpoint are
bound to the spool; changing either requires a new private root. Each tap instance loads its
policy file on every callback. Setting `enabled:false` stops capture immediately
on the next callback and stops the collector on its next pass. The previous
spool remains private for recovery or explicit revocation. Invalid policy returns
a content-free `unavailable` status. The application should expose capture
status to its operator without logging request or response bodies.

## Trusted adapters

The adapter is the same in both modes. In Spark mode, the head's existing
completed-request callback invokes `spark_completed`. In harness mode, the
external client invokes `harness_completed` after receiving the successful
completion, including the last chunk for streaming. Failed, canceled and
internal funnel requests are ineligible. Internal status is trusted caller
context. Mark collector/extraction calls internal so intake never feeds itself.

```python
from glm_rsi_dream.predream.tap import TrafficTap

# root and policy_path are operator-owned local paths.
tap = TrafficTap(root, policy_path=policy_path)
status = tap.harness_completed(
    request,
    principal=authenticated_principal,
    request_id=stable_request_id,
    opt_in=session_capture_consent,
    endpoint=head_endpoint,
    successful=True,
    internal=False,
)
```

Swap the method to `spark_completed` and policy mode to `spark` on the head.
The request ID is stable for retry of the same completion and unique for a new
completion. Callbacks preserve the input object and make no model/network call.
Spool admission is a short SQLite transaction with a 50 ms lock timeout. An I/O
failure returns `unavailable`; original generation remains owned by the caller.
Capture timing and callback overhead under real load are unmeasured.

The separate collector runs one bounded pass:

```sh
python -m glm_rsi_dream.predream.tap_cli \
  --root "$TAP_ROOT" --policy "$TAP_POLICY" \
  collect --endpoint "$HEAD_ENDPOINT" --max-events 4
```

`--watch --interval 5` repeats passes with policy reload, the same durable budgets
and bounded work. It starts no service manager, installs no daemon and performs
no capture by itself. The existing completed-request callback supplies events.
Spark mode runs this collector on the head; harness mode runs it externally and
uses only the head generation API. The worker remains a TP rank with no
API. Use the existing funnel to list, grade and approve the pending rows. The
tap never calls approval, staging, dreaming or training.

## Privacy and limits

Capture accepts only plain text in the final user message. It omits conversation
history, assistant content, system/developer prompts, tool output, headers,
responses, images and audio. Credentials, common contact/financial/health
identifiers, explicit non-retention language, configured denied terms, control
characters and oversized text reject the entire event. Whole-event rejection
keeps retained evidence literal. Conservative regex checks can reject harmless
numbers or words. They cannot identify every secret, obfuscated identifier or
sensitive narrative. Configure denied terms for local context and keep consent
explicit. This is a collection policy, with no universal privacy claim.

The filter runs before source persistence or model access, on every model
response, on extracted claim/evidence/probe fields and before publishing resumed
batches. Raw client tracing is rejected. The tap HTTP client makes one transport attempt
per reserved call, including failures, so transport retry cannot bypass budgets. Errors expose status codes only. Spool
and queue directories use0700; files use0600. This implementation uses POSIX
file locking and SQLite, qualified on macOS CPU. Spark Linux runtime qualification
remains a deployment check.

Default bounds:8192 source bytes,256 retained event receipts,2 MB logical spool
payload,24-hour spool/receipt retention,12 events per sliding minute globally,
4 per principal,24 model calls per sliding minute,24 calls per event,
8 candidates per event and256 pending candidates at tap admission. Intake stops
at capacity. Limits include completed event receipts so draining does not reset
capture budgets. A drain serializes with other tap collectors. The pending queue
cap covers tap admission; independent funnel writers keep their own policy.
A model-rate limit stops a pass until the sliding window opens. Other model,
format or persistence failures isolate that event as `failed`, so later events
continue. Inspect content-free state counts with `inspect`. After correcting
the model, storage or policy blocker, `retry` reopens failed events. Safe source
or prepared batch remains private until retry, revocation or TTL expiry. Errors
retain no raw exception or rejected model answer. An `event_call_limit` requires
raising that explicit per-event budget or changing the model/source recipe
before retry. The operator owns these opening decisions.

The existing queue is append-only and its total historical size is governed by
that funnel, separately from the bounded tap spool.

Stable request IDs are stored as keyed opaque digests; raw IDs and principal
labels stay out of spool records and candidate provenance. Restart preserves
deduplication and budgets. TTL clears spool evidence and retry receipts on a
capture/collector pass. Duplicate replay protection lasts for the configured
retention window. Completed spool evidence is cleared after durable queue
publication. A safe prepared batch is committed before publication; interrupted
publication resumes and skips candidate IDs already in the queue.

Remove a principal from the policy to stop future capture and purge its queued
spool events before further model access. Explicit revocation also prevents
recapture in this spool and purges its event payloads:

```sh
python -m glm_rsi_dream.predream.tap_cli --root "$TAP_ROOT" \
  revoke --principal "$PRINCIPAL"
```

Revocation is permanent for that spool. New consent uses a new private root.
To inspect or retry after repair:

```sh
python -m glm_rsi_dream.predream.tap_cli --root "$TAP_ROOT" inspect
python -m glm_rsi_dream.predream.tap_cli --root "$TAP_ROOT" retry
```

It does not erase candidates already published in the separately governed
append-only funnel queue. Removing those requires the funnel owner's explicit
data-erasure workflow. No revocation logs contain the raw principal label.

## Offline replay

Replay enables only the fixture root. For example, create an offline policy with
`enabled:true`, `mode:harness`, `endpoint:http://recorded.invalid:8888` and
`principals:[recorded-fixture]`, keeping the remaining bounds from defaults.
Then run:

```sh
python -m glm_rsi_dream.predream.tap_cli \
  --root "$REPLAY_ROOT" --policy "$REPLAY_POLICY" \
  replay tests/predream/fixtures/tap_recorded.jsonl \
  --principal recorded-fixture --opt-in
```

The CLI has no network path during replay. Responses come from exact recorded
non-system exchanges and a recognized extraction/prior/comparison protocol. A
missing exchange fails visibly. Source prompts become observed fixture events;
replay grants apply only to these test fixtures. This proves software intake
with historical GLM outputs, separately from fresh generation or live tap
installation. See `tap-recording-provenance.md` for the preserved recording and
historical comparison-instruction difference.
