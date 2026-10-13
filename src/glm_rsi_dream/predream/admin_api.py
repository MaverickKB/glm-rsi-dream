"""Small loopback-only HTTP adapter for the RSI Dreambox admin library."""

import argparse
import hmac
import json
import math
import os
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..files import invalid_constant
from .admin_schema import OPERATIONS, openapi_spec
from .admin_store import AdminError, AdminStore
from .control import initialize_key

MAX_BODY = 64 * 1024 * 1024


def read_token(path: Any) -> Any:
    path = Path(path)
    metadata = path.stat()
    if metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
        raise ValueError("API token must be owned by the host user and private (mode0600)")
    token = path.read_text().strip()
    if len(token) < 32 or re.fullmatch(r"[A-Za-z0-9_-]+", token) is None:
        raise ValueError("API token must contain at least 32 safe characters")
    return token


class AdminHandler(BaseHTTPRequestHandler):
    server: Any
    def log_message(self, *_: Any) -> Any:
        # Requests may contain private knowledge. The host owns access logging.
        pass

    def respond(self, status: Any, value: Any) -> Any:
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        origin = self.headers.get("Origin")
        if origin and origin in self.server.allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE")
        self.end_headers()
        self.wfile.write(raw)

    def handle_request(self) -> Any:
        self.connection.settimeout(10)
        try:
            origin = self.headers.get("Origin")
            if origin and origin not in self.server.allowed_origins:
                raise AdminError(403, "Origin is not allowed")
            if self.command == "OPTIONS":
                self.respond(200, {"allowed": True})
                return
            target = urlsplit(self.path)
            if target.query:
                raise AdminError(400, "Query parameters are unsupported; tokens belong in headers")
            if self.command == "GET" and target.path == "/openapi.json":
                self.respond(200, openapi_spec())
                return
            authorization = self.headers.get("Authorization", "")
            expected = "Bearer " + read_token(self.server.token_file)
            if not hmac.compare_digest(authorization.encode(), expected.encode()):
                raise AdminError(401, "Admin bearer authentication required")
            matched = None
            identifier = None
            for method, route, operation, schema, success in OPERATIONS:
                pattern = "/api/v1" + route.replace("{id}", "([a-z0-9][a-z0-9_-]{0,63})")
                match = re.fullmatch(pattern, target.path)
                if method.upper() == self.command and match is not None:
                    matched = (operation, schema, success)
                    if match.groups():
                        identifier = match.group(1)
                    break
            if matched is None:
                raise AdminError(404, "Route was not found")
            operation, schema, success = matched
            self.server.store.ensure_available(operation)
            body = None
            length = self.headers.get("Content-Length", "0")
            if not length.isdigit() or int(length) > self.server.max_body_bytes:
                raise AdminError(413, "Request body exceeds the allowed size")
            if self.headers.get("Transfer-Encoding"):
                raise AdminError(400, "Chunked request bodies are unsupported")
            if schema is not None:
                if self.headers.get_content_type() != "application/json":
                    raise AdminError(415, "Request body must be application/json")
                if int(length) == 0:
                    raise AdminError(400, "JSON body is required")
                body = json.loads(self.rfile.read(int(length)), parse_constant=invalid_constant)
            elif int(length):
                raise AdminError(400, "This operation has no request body")
            result = self.server.store.request(operation, body, identifier)
            self.respond(success, result)
        except AdminError as error:
            self.respond(error.status, {"error": str(error)})
        except (ValueError, KeyError, TypeError, UnicodeError) as error:
            status = 409 if "revision changed" in str(error).lower() else 400
            self.respond(status, {"error": str(error)})
        except OSError:
            self.respond(503, {"error": "Local admin storage or authority is unavailable"})
        except RuntimeError:
            self.respond(503, {"error": "Host GLM request failed; prepared intake can be retried"})

    do_GET = handle_request
    do_POST = handle_request
    do_PUT = handle_request
    do_PATCH = handle_request
    do_DELETE = handle_request
    do_OPTIONS = handle_request


def create_server(root: Any, token_file: Any, *, host: Any = "127.0.0.1", port: Any = 8789, allowed_origins: Any = (), max_body_bytes: Any = MAX_BODY,
                  client: Any = None, endpoint: Any = "http://127.0.0.1:8888", model: Any = "GLM-5.3-Flash-EXL3", timeout: Any = 120, dream_features_enabled: Any = False) -> Any:
    from .extract import GLMClient

    if host != "127.0.0.1":
        raise ValueError("Development admin server binds IPv4 loopback only")
    read_token(token_file)
    if type(max_body_bytes) is not int or max_body_bytes <= 0:
        raise ValueError("Request body limit must be a positive byte count")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("GLM timeout must be positive and finite")
    for origin in allowed_origins:
        parsed = urlsplit(origin)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.path or parsed.query or parsed.fragment:
            raise ValueError("CORS origins must be exact http(s) origins")
    if client is None:
        client = GLMClient(endpoint, model, timeout=timeout, trace_path=Path(root) / "manual-glm-trace.jsonl")
    store = AdminStore(root, client=client, dream_features_enabled=dream_features_enabled)
    server: Any = HTTPServer((host, port), AdminHandler)
    server.token_file = Path(token_file)
    server.allowed_origins = set(allowed_origins)
    server.max_body_bytes = max_body_bytes
    server.store = store
    return server


def main(argv: Any = None) -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    token = commands.add_parser("init-token")
    token.add_argument("file")
    serve = commands.add_parser("serve")
    serve.add_argument("--root", required=True)
    serve.add_argument("--token-file", required=True)
    serve.add_argument("--port", type=int, default=8789)
    serve.add_argument("--allow-origin", action="append", default=[])
    serve.add_argument("--max-body-bytes", type=int, default=MAX_BODY)
    serve.add_argument("--enable-dream-features", action="store_true", help="Host opens development plans after pocket-OFF qualification")
    serve.add_argument("--endpoint", default="http://127.0.0.1:8888", help="Host-owned GLM API origin")
    serve.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    serve.add_argument("--timeout", type=float, default=120, help="Seconds per GLM request")
    args = parser.parse_args(argv)
    if args.command == "init-token":
        initialize_key(args.file)
        print(json.dumps({"token_file": args.file, "created": True}))
        return 0
    server = create_server(args.root, args.token_file, port=args.port, allowed_origins=args.allow_origin,
                           max_body_bytes=args.max_body_bytes, endpoint=args.endpoint, model=args.model, timeout=args.timeout,
                           dream_features_enabled=args.enable_dream_features)
    print(json.dumps({"listening": "http://127.0.0.1:" + str(server.server_port)}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
