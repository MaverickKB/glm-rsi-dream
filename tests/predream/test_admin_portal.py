"""CPU checks for static packaging and transparent local development transport."""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from glm_rsi_dream.predream.admin_portal import create_server


@contextmanager
def serving(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_port)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def request(url, method="GET", body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = Request(url, data=data, method=method, headers=headers or {})
    try:
        response = urlopen(req, timeout=5)
    except HTTPError as error:
        response = error
    with response:
        return response.status, response.read(), dict(response.headers)


def test_packaged_static_preview_and_explicit_demo():
    with serving(create_server()) as url:
        status, content, headers = request(url + "/")
        assert status == 200
        assert b"Simple" in content and b"Advanced" in content
        assert b"<title>RSI Dreambox</title>" in content
        assert b'aria-label="RSI Dreambox, Rules"' in content
        assert b'img class="brand-symbol" src="favicon.png"' in content
        assert b'link rel="icon" href="favicon.png"' in content
        assert b"Learning workspace" not in content
        assert "script-src 'self'" in headers["Content-Security-Policy"]
        for path in (
            "portal.css",
            "portal.js",
            "demo.json",
            "help.json",
            "favicon.png",
            "hero.png",
            "openapi.json",
        ):
            assert request(url + "/" + path)[0] == 200
        status, content, _ = request(url + "/api/v1/queue")
        assert status == 503 and b"--api" in content
        assert request(url + "/../pyproject.toml")[0] == 404
        assert request(url + "/portal.js", "POST", {})[0] == 405


def test_preview_request_limit_is_explicit_and_configurable():
    with serving(create_server(api_url="http://127.0.0.1:1", max_body_bytes=16)) as url:
        assert (
            request(
                url + "/api/v1/pockets/import", "POST", {"package": "long payload"}
            )[0]
            == 413
        )
    with pytest.raises(ValueError, match="positive"):
        create_server(max_body_bytes=0)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.test",
        "http://192.0.2.1",
        "http://user:password@localhost",
        "http://localhost/private",
        "http://localhost?token=fixture",
    ],
)
def test_preview_refuses_nonlocal_or_credentialed_proxy_origins(url):
    with pytest.raises(ValueError, match="loopback"):
        create_server(api_url=url)


def test_proxy_preserves_gate_method_payload_and_origin():
    observed = []

    class Backend(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_PATCH(self):
            authorized = (
                self.headers.get("Authorization") == "Bearer fictional-api-token"
            )
            observed.append(
                {
                    "method": self.command,
                    "origin": self.headers.get("Origin"),
                    "body": self.rfile.read(int(self.headers["Content-Length"])),
                }
            )
            raw = json.dumps({"authorized": authorized}).encode()
            self.send_response(200 if authorized else 401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw)

    with serving(ThreadingHTTPServer(("127.0.0.1", 0), Backend)) as backend:
        with serving(create_server(api_url=backend)) as portal:
            body = {
                "expected_revision": "a" * 64,
                "hold": True,
                "provenance": {"version": 1.0},
            }
            headers = {"Content-Type": "application/json", "Origin": portal}
            assert (
                request(portal + "/api/v1/queue/fixture", "PATCH", body, headers)[0]
                == 401
            )
            headers["Authorization"] = "Bearer fictional-api-token"
            status, result, _ = request(
                portal + "/api/v1/queue/fixture", "PATCH", body, headers
            )
            assert status == 200 and json.loads(result)["authorized"]
            assert observed[-1]["method"] == "PATCH"
            assert observed[-1]["origin"] == portal
            assert json.loads(observed[-1]["body"]) == body
            assert observed[-1]["body"] == json.dumps(body).encode()


def test_proxy_rejects_redirects_without_forwarding_credentials():
    class Backend(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", "http://example.invalid/private")
            self.end_headers()

    with serving(ThreadingHTTPServer(("127.0.0.1", 0), Backend)) as backend:
        with serving(create_server(api_url=backend)) as portal:
            assert (
                request(
                    portal + "/api/v1/queue",
                    headers={"Authorization": "Bearer fictional-api-token"},
                )[0]
                == 502
            )
