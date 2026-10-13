"""Temporary loopback preview for the bundled static RSI Dreambox portal.

Run with --api to proxy an absorb-owned local development API. Authorization
passes through unchanged. This preview owns no authority or model operations.
"""

import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/portal.css": ("portal.css", "text/css; charset=utf-8"),
    "/portal.js": ("portal.js", "text/javascript; charset=utf-8"),
    "/demo.json": ("demo.json", "application/json"),
    "/help.json": ("help.json", "application/json"),
    "/favicon.png": ("favicon.png", "image/png"),
    "/hero.png": ("hero.png", "image/png"),
}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: Any, msg: Any, headers: Any, newurl: Any) -> Any:
        raise URLError("Development API redirects are unsupported")


def create_server(port: Any = 0, api_url: Any = None, max_body_bytes: Any = 64 * 1024 * 1024, api_timeout: Any = 900, *, access: Any = None, bind: Any = "127.0.0.1") -> Any:
    """Construct a loopback server, without starting any thread or connection."""
    if type(max_body_bytes) is not int or max_body_bytes <= 0:
        raise ValueError("Preview request limit must be a positive integer")
    if type(api_timeout) not in (int, float) or not math.isfinite(api_timeout) or api_timeout <= 0:
        raise ValueError("Preview API timeout must be positive and finite")
    if api_url:
        target = urlsplit(api_url)
        if (
            target.scheme != "http"
            or target.hostname not in ("127.0.0.1", "localhost", "::1")
            or target.username
            or target.password
            or target.path not in ("", "/")
            or target.query
            or target.fragment
        ):
            raise ValueError("Preview API must be a loopback HTTP origin")
        api_url = api_url.rstrip("/")
    assets = files("glm_rsi_dream").joinpath("resources/admin_portal")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: Any, *args: Any) -> Any:
            return

        def send_content(self, status: Any, content: Any, content_type: Any, extra_headers: Any = None) -> Any:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; img-src 'self' data:; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(content)

        def json_error(self, status: Any, message: Any) -> Any:
            content = json.dumps({"error": message}).encode()
            self.send_content(status, content, "application/json")

        def dispatch(self) -> Any:
            path = urlsplit(self.path).path
            if access is not None and access.handle(self, path):
                return
            if path.startswith("/api/") or path == "/openapi.json" and api_url:
                self.proxy()
                return
            if self.command != "GET":
                self.json_error(405, "Static preview accepts GET only")
                return
            if path == "/openapi.json":
                resource = files("glm_rsi_dream").joinpath(
                    "resources/rsi-admin-openapi.json"
                )
                if resource.is_file():
                    self.send_content(200, resource.read_bytes(), "application/json")
                else:
                    self.json_error(503, "Admin API specification is not packaged yet")
                return
            if path not in ASSETS:
                self.json_error(404, "Unknown portal resource")
                return
            filename, content_type = ASSETS[path]
            self.send_content(200, assets.joinpath(filename).read_bytes(), content_type)

        def proxy(self) -> Any:
            if not api_url:
                self.json_error(503, "Connect a local admin API with --api")
                return
            if self.headers.get("Transfer-Encoding"):
                self.json_error(400, "Chunked preview requests are unsupported")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.json_error(400, "Invalid content length")
                return
            if length < 0 or length > max_body_bytes:
                self.json_error(413, "Preview request is too large")
                return
            body = self.rfile.read(length) if length else None
            headers = {}
            for name in ("Authorization", "Content-Type", "Accept", "Origin"):
                if self.headers.get(name):
                    headers[name] = self.headers[name]
            if access is not None:
                headers["Authorization"] = "Bearer " + access.api_token
                if access.api_origin:
                    headers.pop("Origin", None)
            target_url = access.api_origin if access is not None and access.api_origin else api_url
            request = Request(target_url + self.path, data=body, headers=headers, method=self.command)
            try:
                with build_opener(NoRedirect()).open(request, timeout=api_timeout) as response:
                    self.send_content(
                        response.status,
                        response.read(),
                        response.headers.get("Content-Type", "application/json"),
                    )
            except HTTPError as error:
                self.send_content(
                    error.code,
                    error.read(),
                    error.headers.get("Content-Type", "application/json"),
                )
            except URLError:
                self.json_error(502, "Local admin API is unavailable")

        do_GET = dispatch
        do_POST = dispatch
        do_PUT = dispatch
        do_PATCH = dispatch
        do_DELETE = dispatch

    server = ThreadingHTTPServer((bind, port), Handler)
    if access is not None:
        access.set_port(server.server_port)
    return server


def main(argv: Any = None) -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument(
        "--api", help="Loopback admin API origin, such as http://127.0.0.1:8768"
    )
    parser.add_argument("--max-body-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--api-timeout", type=float, default=900, help="Seconds for a GLM-backed API operation")
    args = parser.parse_args(argv)
    server = create_server(args.port, args.api, args.max_body_bytes, args.api_timeout)
    print(f"RSI Dreambox preview: http://127.0.0.1:{server.server_port}", flush=True)
    print("Explicit fictional demo: add ?demo=1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
