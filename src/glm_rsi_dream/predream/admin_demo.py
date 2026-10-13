"""Run RSI Dreambox locally with explicit fictional CPU fixtures.

Each workspace is fresh. Protocol responses are scripted fixtures, not GLM
judgments. Temporary credentials expire with this process and are removed on exit.
"""

import argparse
import json
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..files import write_json
from . import control
from .absorb import stage
from .admin_api import create_server as create_api
from .admin_api import read_token
from .admin_portal import create_server as create_portal
from .admin_store import AdminStore
from .grade import grade_pending
from .manual import learn_file

FIXTURE_FACTS = {
    "Fictional Aster station's hatch is cobalt.": "pocket",
    "Fictional Iona has a meeting Friday evening.": "memory",
    "Fictional Aster station's dock is silver.": "pocket",
    "Fictional Aster station's beacon is amber.": "pocket",
}


class FixtureGLM:
    """Declared CPU fixture replies, never a production model or fallback."""

    def __init__(self, root: Any) -> None:
        self.root = Path(root)
        self.calls = 0

    def chat(self, messages: Any) -> Any:
        self.calls += 1
        if len(messages) == 2 and messages[0]["role"] == "system":
            rows = []
            for text in messages[1]["content"].splitlines():
                if text in FIXTURE_FACTS:
                    rows.append({"text": text, "prompt_evidence": text,
                                 "probe_question": "Declared fictional fixture question"})
            response: dict[str, Any] = {"candidates": rows}
        else:
            request = messages[0]["content"]
            payload = None
            for line in request.splitlines():
                if line.startswith("{"):
                    payload = json.loads(line)
                    break
            if payload is None:
                raise ValueError("CPU fixture requires a declared protocol payload")
            text = payload["candidate"]
            declared = text in FIXTURE_FACTS
            if request.startswith("Review proposed knowledge"):
                response = {"safe": declared, "kind": "new", "targets": [],
                            "reason": "Declared fictional CPU fixture review"}
            else:
                route = FIXTURE_FACTS.get(text, "memory")
                response = {"verdict": "accept" if declared else "reject", "route": route,
                            "pocket": "aster" if route == "pocket" else None,
                            "reason": "Declared fictional CPU fixture, no real GLM judgment", "score": 0.95}
        write_json(self.root / "demo/live-fixture-calls.json", {"synthetic_protocol_calls": self.calls,
                    "model_calls": 0, "latest_fixture_response": response})
        return json.dumps(response)


@contextmanager
def demo_authority(root: Any, mode: Any) -> Any:
    """Scope a temporary demo grant to one root and restore the caller's environment."""
    if mode not in ("spark", "harness"):
        raise ValueError("Demo mode must be spark or harness")
    with tempfile.TemporaryDirectory(prefix="rsi-admin-demo-") as private_directory:
        private = Path(private_directory)
        key = private / "signing.key"
        token = private / "http.token"
        permission = private / "permission.json"
        control.initialize_key(key)
        control.initialize_key(token)
        settings = {"PREDREAM_ADMIN_KEY_FILE": str(key),
                    "PREDREAM_ADMIN_PERMISSION_FILE": str(permission),
                    "PREDREAM_MODE": mode, "PREDREAM_AUTONOMY": "manual"}
        previous = {}
        for name, value in settings.items():
            previous[name] = os.environ.get(name)
            os.environ[name] = value
        try:
            control.issue_permission(permission, admin_id="fixture:portal-demo", root_scope=root,
                                     modes=[mode], autonomy=["manual", "supervised", "automatic"])
            yield {"token_file": token, "permission_file": permission}
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def prepare_fixture(root: Any, *, dream_features_enabled: Any = False) -> Any:
    """Exercise prepared manual intake and the ordinary review/grade/controller path."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    statements = list(FIXTURE_FACTS)
    rows = []
    for text in statements:
        rows.append({"statement": text})
    pack = root / "demo/prepared-knowledge.json"
    write_json(pack, {"facts": rows})
    responses = []
    for route in ("pocket", "memory", "pocket", "pocket"):
        responses.append({"safe": True, "kind": "new", "targets": [],
                          "reason": "Explicit fictional CPU safety/conflict response."})
        responses.append({"verdict": "accept", "route": route,
                          "pocket": "aster" if route == "pocket" else None,
                          "reason": "Explicit fictional CPU routing response.", "score": 0.95})
    replies = iter(responses)

    def fixture_client(_: Any) -> Any:
        return json.dumps(next(replies))

    intake = learn_file(root, pack, client=fixture_client, source_ref="fictional:admin-portal-demo")
    graded = grade_pending(root, fixture_client)
    if graded["errors"] or len(graded.get("awaiting_approval", [])) != 4:
        raise ValueError("Fixture protocol did not produce four manual proposals")
    identifiers = []
    for row in intake["queued"]:
        identifiers.append(row["id"])
    control.approve(root, [identifiers[3]])
    stage(root, "fixture-stage")
    store = AdminStore(root, dream_features_enabled=dream_features_enabled)
    store.request("rules_preset", {"id": "companion", "preset": "companion"})
    store.request("pockets_create", {"id": "aster", "topic": "aster", "capacity": 1})
    if dream_features_enabled:
        schedule = store.request("schedule_get")
        store.request("schedule_put", {"expected_revision": schedule["revision"], "enabled": True,
            "timezone": "America/New_York", "windows": [{"days": [0, 1, 2, 3, 4, 5, 6],
            "start": "19:00", "end": "23:59"}], "min_idle_seconds": 900, "max_active_requests": 0})
        store.request("dreams_create", {"id": "fixture-dream", "run_id": "fixture-stage",
                                      "not_before": "2026-10-13T19:00:00-04:00"})
    result = {"kind": "fictional_cpu_demo", "judgments": "synthetic_fixture_responses",
              "model_calls": 0, "training_executed": False, "tap_enabled": False,
              "fixture_protocol_responses": len(responses), "staged_run": "fixture-stage",
              "dream_features_enabled": dream_features_enabled,
              "candidate_ids": {"pocket": identifiers[0], "memory": identifiers[1],
                                "veto": identifiers[2], "staged": identifiers[3]}}
    write_json(root / "demo/fixture-receipt.json", result)
    return result


def main(argv: Any = None) -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Fresh local demo workspace")
    parser.add_argument("--mode", choices=("spark", "harness"), default="harness")
    parser.add_argument("--port", type=int, default=8767, help="Loopback portal port, 0 chooses a free port")
    parser.add_argument("--enable-dream-features", action="store_true",
                        help="Open manual dream fixtures only after the owner's recall qualification")
    parser.add_argument("--advanced-api-preview", action="store_true",
                        help="Use an explicit browser API credential for Advanced transport development")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    with demo_authority(root, args.mode) as authority:
        fixture = prepare_fixture(root, dream_features_enabled=args.enable_dream_features)
        api = create_api(root, authority["token_file"], port=0, client=FixtureGLM(root),
                         dream_features_enabled=args.enable_dream_features)
        try:
            api_url = "http://127.0.0.1:" + str(api.server_port)
            access = None
            if not args.advanced_api_preview:
                from .admin_host import HostAccess

                password_file = Path(authority["token_file"]).parent / "password.json"
                access = HostAccess(authority["token_file"], password_file, "http://127.0.0.1:0")
            portal = create_portal(args.port, api_url, access=access)
            portal_url = "http://127.0.0.1:" + str(portal.server_port)
            api.allowed_origins.add(portal_url)
            thread = threading.Thread(target=api.serve_forever, kwargs={"poll_interval": 0.05})
            thread.start()
            ready = {"portal": portal_url, "api": api_url, "mode": args.mode,
                     "hosted": access is not None, "fixture": fixture}
            if args.advanced_api_preview:
                ready["credential"] = read_token(authority["token_file"])
            print(json.dumps(ready), flush=True)
            try:
                portal.serve_forever(poll_interval=0.05)
            except KeyboardInterrupt:
                pass
            finally:
                portal.server_close()
                api.shutdown()
                thread.join(timeout=5)
        finally:
            api.server_close()


if __name__ == "__main__":
    main()
