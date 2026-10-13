# System

Check the connection, execution mode and current capabilities. Start here when an action cannot proceed.

## Simple

### Open the portal

Set one password on the first visit, then sign in once. Your browser keeps an HttpOnly session cookie. The engine credential stays on the host. A valid session covers navigation and changes without another password prompt.

### Choose a session timeout

Security in System or Portal settings offers 15 min, 1 hour, 8 hours and 1 day. The timeout applies to new sign-ins. Activity never extends the original expiry. Sign out other sessions keeps this browser signed in.

### Resolve a failed save

Sign in when your session expires. An expired host grant can show one Approve action next to the requested level. For a changed item, refresh its current values and review your edit.

## Advanced

### Inspect values and API

Open the raw values and API view to compare responses with /openapi.json. Export a policy for a fresh run when changing experiment settings. Host-owned runtime values remain visible according to the connected API's capabilities.

### Check collection boundaries

Traffic collection stays off and undeployed during development. The future collection design requires per-user exclusions and an approved time window. Enabling a rule or schedule only changes its own planning state.

### Security and another engine

Advanced offers a custom timeout from 5 min to 1 day and can turn the portal lock off after a short warning. Connect to an engine accepts its URL and key once; the host saves them privately and proxies requests on the same origin.

## Settings and details

### Engine connection

The default portal runs beside the engine on the Spark head. Advanced can select another engine by URL and API key. The host stores that target privately and proxies API requests; the browser stays on the same origin.

### Engine API key

The host loads its private mode0600 engine credential. Simple uses the portal password and session instead. Advanced Connect to an engine can accept a separate key, which the host saves privately. Keep keys out of shared exports and URLs.

### Execution mode

Harness runs the portable funnel outside the engine host; spark runs it beside the engine. The host-controlled mode must match the policy. Selecting a mode describes where the process runs; relocating it is an operator action.

### Model

The generation model name supplied to the funnel client. It is a runtime argument, separate from policy fields. Verify the connected host's current identity before a native operation.

### Generation endpoint

The head-owned chat endpoint used by the funnel client. The client appends /v1 when needed. The TP=2 worker participates in serving and has no generation API.

### Request timeout

The host sets seconds allowed per model request. Extraction CLI defaults to 180; the admin API defaults to 120 through --timeout. The local portal proxy allows 900 seconds per API operation through --api-timeout (positive and finite), covering bounded model protocol retries. A timeout remains visible; the same source can be retried without duplicate queue insertion.

### Traffic collection design

Collection is OFF by default. Enabling capture requires explicit owner approval, an approved window, trusted per-request consent and configured per-user exclusions. Rule and schedule editors keep their own planning scope.

### Demo mode

Fictional data and edits live in browser memory. The portal labels Demo explicitly. Connect to the API to work with its stored data and authority.

### API errors

400 means invalid input;401 requires sign-in or a valid Advanced connection;403 means the requested action or origin was denied;404 means the item or route is missing;409 means changed or conflicting data;429 means wait before another sign-in attempt. Refresh current data before resolving a conflict.

### Keyboard controls

Outside an input or open panel, A switches Simple and Advanced. In Queue, / focuses search. Escape closes a panel. In Advanced Queue, Ctrl+Enter or Command+Enter reviews selected rows for bulk approval.

### Host API settings

The backend owner selects a fixed storage root at startup and exact allowed browser origins with --allow-origin. Maximum request bytes is a positive host setting, default 67108864 (64 MiB), changed with --max-body-bytes. HTTP 413 means the request exceeded that limit. The portal preserves the host boundary.

### Portal security

The lock is on by default. One password uses salted scrypt storage. The server persists hashes of random session tokens. Cookies are HttpOnly and SameSite=Strict, with Secure underHTTPS. One session_seconds value defaults to86400 and validates300..86400. New logins use the setting; existing login and expiry times remain unchanged by activity or settings edits. Logout, password change and Sign out other sessions revoke the appropriate sessions. Host and Origin checks protect changes; login attempts are rate-limited.

[All pages](README.md)
