"""Temporary local portal/API proof with explicitly fictional GLM replies.

Run from the checkout: PYTHONPATH=src python examples/predream/manual_cpu_demo.py
The temporary workspace and private fixture credentials are removed on exit.
This opens loopback servers only and never contacts a generation endpoint.
"""

import json
import os
import tempfile
import threading
from pathlib import Path

from glm_rsi_dream.predream import control
from glm_rsi_dream.predream.admin_api import create_server as api_server
from glm_rsi_dream.predream.admin_portal import create_server as portal_server

FIXTURE_FACTS = {
    "Fictional Lumen has three sample bays.": "pocket",
    "Fictional Mira departs tonight.": "memory",
    "Fictional Orin's beacon is teal.": "weights",
}


class FixtureGLM:
    """A closed CPU test fixture, never a semantic or production model fallback."""

    def __init__(self):
        self.calls = 0
        self.review_failures = 3

    def chat(self, messages):
        self.calls += 1
        if len(messages) == 2 and messages[0]["role"] == "system":
            rows = []
            for text in messages[1]["content"].splitlines():
                if text in FIXTURE_FACTS:
                    rows.append({"text": text, "prompt_evidence": text, "probe_question": "Fictional fixture question"})
            return json.dumps({"candidates": rows})
        request = messages[0]["content"]
        payload = None
        for line in request.splitlines():
            if line.startswith("{"):
                payload = json.loads(line)
                break
        if payload is None:
            raise ValueError("CPU fixture requires a model protocol payload")
        text = payload["candidate"]
        if request.startswith("Review proposed knowledge"):
            if text == "Fictional Orin's beacon is teal." and self.review_failures:
                self.review_failures -= 1
                return "Deliberately invalid CPU fixture reply"
            return json.dumps({"safe": True, "kind": "new", "targets": [],
                               "reason": "Closed fictional CPU fixture review"})
        route = FIXTURE_FACTS.get(text, "memory")
        verdict = "accept" if text in FIXTURE_FACTS else "reject"
        return json.dumps({"verdict": verdict, "route": route, "pocket": "lumen" if route == "pocket" else None,
                           "reason": "Closed fictional CPU fixture, no real GLM judgment", "score": 0.95})


def main():
    with tempfile.TemporaryDirectory(prefix="prepared-knowledge-cpu-") as directory:
        directory = Path(directory)
        root = directory / "workspace"
        key = directory / "signing.key"
        token = directory / "http.token"
        permission = directory / "permission.json"
        control.initialize_key(key)
        control.initialize_key(token)
        os.environ["PREDREAM_ADMIN_KEY_FILE"] = str(key)
        os.environ["PREDREAM_ADMIN_PERMISSION_FILE"] = str(permission)
        os.environ["PREDREAM_MODE"] = "harness"
        os.environ["PREDREAM_AUTONOMY"] = "manual"
        control.issue_permission(permission, admin_id="fixture:manual-demo", root_scope=root,
                                 modes=["harness"], autonomy=["manual"])
        client = FixtureGLM()
        api = api_server(root, token, port=0, client=client)
        origin = "http://127.0.0.1:" + str(api.server_port)
        portal = portal_server(api_url=origin)
        api.allowed_origins.add("http://127.0.0.1:" + str(portal.server_port))
        threads = []
        for server in (api, portal):
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
            thread.start()
            threads.append(thread)
        print(json.dumps({"fixture": "fictional CPU only", "portal": "http://127.0.0.1:" + str(portal.server_port),
                          "root": str(root), "token_file": str(token), "facts": list(FIXTURE_FACTS)}), flush=True)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            for server in (portal, api):
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join(timeout=5)
            print(json.dumps({"fixture_calls": client.calls, "stopped": True}), flush=True)


if __name__ == "__main__":
    main()
