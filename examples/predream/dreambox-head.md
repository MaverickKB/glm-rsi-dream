# RSI Dreambox head installation

The target is the configured Spark head. RSI Dreambox runs as its own
small process on port 8767. It owns an internal loopback admin API and loads its
private credential from a local file. The GLM container and generation port
remain owned by the engine. This package is built and tested locally. Deployment
waits for explicit owner approval.

Build a wheel from the owning source checkout:

```sh
python -m pip wheel --no-deps . --wheel-dir dist
```

At the approved head installation, install that wheel into a dedicated virtual
environment. Run its `glm-dreambox init` command to write private files and a
service unit. The command does not start or enable a service:

```sh
glm-dreambox init --state-dir /srv/dreambox
```

It creates `host-config.json`, `controller.key`, `api-credential`, a bounded
`permission.json`, and `rsi-dreambox.service`. Credentials belong to the host
user with mode 600. The unit uses the same virtual environment interpreter and
loads the config file. The host admin grant covers this workspace in Spark mode
and the Manual, Supervised and Automatic levels. Expired access can be renewed
through the portal's explicit Approve action within that existing scope.

After deployment approval, the operator may install the emitted unit in the
host's systemd directory. The unit is independent of model serving. A foreground
run uses:

```sh
glm-dreambox serve --config /srv/dreambox/host-config.json
```

The first visit to `http://127.0.0.1:8767` sets one portal password. Subsequent
visits sign in once. Valid sessions carry changes without another password
prompt. Browser requests never contain the head's engine credential. Passwords
are salted scrypt hashes; session files contain random-token hashes and expiry.
The cookie is HttpOnly and SameSite=Strict, and gains Secure when the configured
public origin is HTTPS. Origin and Host checks protect state-changing calls.
Five failed login attempts within one minute pause further attempts.

Settings exposes session presets of 15 minutes, 1 hour, 8 hours and 1 day.
Advanced accepts a custom duration between 300 and 86400 seconds. The sole
setting is `session_seconds`, default 86400. Each login fixes its absolute expiry.
Activity keeps that expiry unchanged. Changing the setting affects new logins;
"Sign out other sessions" keeps the current browser. Logout revokes its server
session. A password change revokes every prior session and signs in the current
browser with the new password. Turning off the lock is an explicit Advanced
action with a short warning; an absent password file always starts locked.

The same-origin session API is documented in
`LOOP2-shared/HEAD-HOSTING.md`. The typed admin API is available at
`/openapi.json` after login. Dream and schedule endpoints remain **Under
development** behind the default-off server flag. Queue, rules and raw-fact
pockets remain usable.

For a local CPU demonstration, run:

```sh
python -m glm_rsi_dream.predream.admin_demo --root /tmp/dreambox-fresh-fixture --mode spark --port 0
```

This uses a fictional prepared pack and eight declared synthetic protocol
replies. Set any temporary demonstration password in the browser. Ctrl+C closes
both listeners and removes the private demo credentials and session files.
Its fixture receipts remain under the selected fresh root. It calls no model
and performs no training or deployment.

A portal on another host connects to an engine through Advanced using its
address and key. Keep Simple on the normal hosted session path. The temporary
API transport preview is explicitly selected with `--advanced-api-preview`.
