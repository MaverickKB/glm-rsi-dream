"""Separate Spark-head RSI Dreambox process with optional portal password."""

import argparse
import hashlib
import hmac
import json
import os
import pwd
import secrets
import sys
import threading
import time
from http.cookies import CookieError, SimpleCookie
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..files import read_json, write_json
from . import control
from .admin_api import create_server as create_api
from .admin_api import read_token
from .admin_portal import create_server as create_portal

COOKIE = "rsi_dreambox_session"
SESSION_SECONDS = 86400


class HostAccess:
    def __init__(self, credential_file: Any, password_file: Any, public_origin: Any) -> None:
        self.api_token = read_token(credential_file)
        self.local_token = self.api_token
        self.api_origin = None
        self.password_file = Path(password_file)
        self.sessions_file = self.password_file.parent / "sessions.json"
        self.connection_file = self.password_file.parent / "connection.json"
        origin = urlsplit(public_origin)
        if origin.scheme not in ("http", "https") or not origin.hostname or origin.username or origin.password:
            raise ValueError("Portal origin must be an HTTP or HTTPS origin")
        if origin.path not in ("", "/") or origin.query or origin.fragment:
            raise ValueError("Portal origin cannot include a path, query or fragment")
        self.origin = public_origin.rstrip("/")
        self.lock = threading.Lock()
        self.login_attempts: dict[Any, Any] = {}
        self.password_settings()
        if self.connection_file.exists():
            connection = self.private_json(self.connection_file)
            self.validate_connection(connection)
            self.api_origin = connection["url"]
            self.api_token = connection["key"]

    def set_port(self, port: Any) -> Any:
        origin = urlsplit(self.origin)
        if origin.port == 0:
            self.origin = origin.scheme + "://" + origin.hostname + ":" + str(port)

    def private_json(self, path: Any) -> Any:
        metadata = path.stat()
        if metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
            raise ValueError("Portal access files must be private and owned by the host user")
        return read_json(path)

    def password_settings(self) -> Any:
        if not self.password_file.exists():
            return {"enabled": True, "hash": None, "salt": None, "session_seconds": SESSION_SECONDS}
        settings = self.private_json(self.password_file)
        if type(settings["enabled"]) is not bool or type(settings["session_seconds"]) is not int:
            raise ValueError("Invalid portal lock settings")
        if not 300 <= settings["session_seconds"] <= SESSION_SECONDS:
            raise ValueError("Session length must be between 300 and 86400 seconds")
        if settings["hash"] is not None:
            if len(bytes.fromhex(settings["salt"])) != 16 or len(bytes.fromhex(settings["hash"])) != 32:
                raise ValueError("Invalid portal password format")
        return settings

    def hash_password(self, password: Any, salt: Any) -> Any:
        return hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=32768, r=8, p=1,
                              maxmem=64 * 1024 * 1024, dklen=32).hex()

    def validate_connection(self, connection: Any) -> Any:
        origin = urlsplit(connection["url"])
        if origin.scheme not in ("http", "https") or not origin.hostname or origin.username or origin.password:
            raise ValueError("Use an HTTP or HTTPS engine address")
        if origin.path not in ("", "/") or origin.query or origin.fragment:
            raise ValueError("Use an engine address without a path")
        if origin.port == 0:
            raise ValueError("Engine port must be greater than zero")
        key = connection["key"]
        if not isinstance(key, str) or len(key) < 32 or not key.isascii():
            raise ValueError("Engine key format is invalid")
        for character in key:
            if not character.isalnum() and character not in "_-":
                raise ValueError("Engine key format is invalid")

    def session_hash(self, handler: Any) -> Any:
        try:
            cookies = SimpleCookie(handler.headers.get("Cookie", ""))
            value = cookies[COOKIE].value
            if len(value) != 64:
                return None
            bytes.fromhex(value)
            return hashlib.sha256(value.encode()).hexdigest()
        except (CookieError, KeyError, ValueError):
            return None

    def sessions(self) -> Any:
        if not self.sessions_file.exists():
            return {}
        return self.private_json(self.sessions_file)

    def cookie(self, token: Any, settings: Any) -> Any:
        result = COOKIE + "=" + token + "; Path=/; HttpOnly; SameSite=Strict; Max-Age="
        result += str(settings["session_seconds"])
        if self.origin.startswith("https://"):
            result += "; Secure"
        return result

    def authenticate(self, handler: Any, settings: Any) -> Any:
        if not settings["enabled"]:
            return True
        if settings["hash"] is None:
            return False
        session_hash = self.session_hash(handler)
        sessions = self.sessions()
        if session_hash not in sessions or sessions[session_hash]["expires_at"] <= time.time():
            return False
        return True

    def new_session(self, settings: Any) -> Any:
        token = secrets.token_hex(32)
        sessions = self.sessions()
        session_hash = hashlib.sha256(token.encode()).hexdigest()
        logged_in = time.time()
        sessions[session_hash] = {"login_at": logged_in, "expires_at": logged_in + settings["session_seconds"]}
        write_json(self.sessions_file, sessions)
        return self.cookie(token, settings)

    def status(self, settings: Any, authenticated: Any) -> Any:
        return {"hosted": True, "authenticated": authenticated, "password_enabled": settings["enabled"],
                "setup_required": settings["enabled"] and settings["hash"] is None,
                "session_seconds": settings["session_seconds"]}

    def respond(self, handler: Any, status: Any, value: Any, cookie: Any = None) -> Any:
        headers = {"Set-Cookie": cookie} if cookie else None
        handler.send_content(status, json.dumps(value).encode(), "application/json", headers)

    def json_body(self, handler: Any, field: Any) -> Any:
        if handler.headers.get("Transfer-Encoding"):
            raise ValueError("Unsupported request encoding")
        if handler.headers.get("Content-Type", "").split(";")[0] != "application/json":
            raise ValueError("Use a JSON request")
        length = int(handler.headers.get("Content-Length", "0"))
        if not 0 < length <= 8192:
            raise ValueError("Invalid request size")
        value = json.loads(handler.rfile.read(length))
        required = {field} if isinstance(field, str) else set(field)
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("Request fields are invalid")
        if isinstance(field, str):
            return value[field]
        return value

    def password_body(self, handler: Any) -> Any:
        password = self.json_body(handler, "password")
        if not isinstance(password, str) or len(password) > 1024:
            raise ValueError("Password must be text up to 1024 characters")
        return password

    def handle(self, handler: Any, path: Any) -> Any:
        if handler.headers.get("Host") != urlsplit(self.origin).netloc:
            handler.json_error(421, "Use the configured portal address")
            return True
        origin = handler.headers.get("Origin")
        if origin and origin != self.origin or handler.command != "GET" and origin != self.origin:
            handler.json_error(403, "Use this portal to make changes")
            return True
        try:
            with self.lock:
                settings = self.password_settings()
                authenticated = self.authenticate(handler, settings)
                setup = settings["enabled"] and settings["hash"] is None
                if path == "/session/status" and handler.command == "GET":
                    self.respond(handler, 200, self.status(settings, authenticated))
                    return True
                if path == "/session/login" and handler.command == "POST":
                    address = handler.client_address[0]
                    attempts = self.login_attempts.get(address, [])
                    recent = []
                    for attempted_at in attempts:
                        if attempted_at > time.time() - 60:
                            recent.append(attempted_at)
                    if len(recent) >= 5:
                        handler.json_error(429, "Try again in a minute")
                        return True
                    password = self.password_body(handler)
                    if settings["enabled"]:
                        if setup or not hmac.compare_digest(self.hash_password(password, settings["salt"]), settings["hash"]):
                            recent.append(time.time())
                            self.login_attempts[address] = recent
                            handler.json_error(401, "Password did not match")
                            return True
                    self.login_attempts.pop(address, None)
                    cookie = self.new_session(settings) if settings["enabled"] else None
                    self.respond(handler, 200, self.status(settings, True), cookie)
                    return True
                first_setup = setup and path == "/session/password" and handler.command == "PUT"
                if not authenticated and not first_setup and path not in ("/portal.js", "/portal.css", "/favicon.png", "/hero.png"):
                    if path in ("/", "/index.html") and handler.command == "GET":
                        content = files("glm_rsi_dream").joinpath("resources/admin_portal/index.html").read_bytes()
                        handler.send_content(401, content, "text/html; charset=utf-8")
                    else:
                        handler.json_error(401, "Enter your portal password")
                    return True
                if path == "/session/password" and handler.command == "PUT":
                    password = self.password_body(handler)
                    if setup and not password:
                        raise ValueError("Set a password to open the portal")
                    settings = dict(settings, enabled=bool(password), hash=None, salt=None)
                    if password:
                        settings["salt"] = secrets.token_hex(16)
                        settings["hash"] = self.hash_password(password, settings["salt"])
                    write_json(self.password_file, settings)
                    write_json(self.sessions_file, {})
                    cookie = self.new_session(settings) if password else COOKIE + "=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0"
                    self.respond(handler, 200, self.status(settings, True), cookie)
                    return True
                if path == "/session/settings" and handler.command == "PUT":
                    seconds = self.json_body(handler, "session_seconds")
                    if type(seconds) is not int or not 300 <= seconds <= SESSION_SECONDS:
                        raise ValueError("Session length must be between 300 and 86400 seconds")
                    settings["session_seconds"] = seconds
                    write_json(self.password_file, settings)
                    authenticated = self.authenticate(handler, settings)
                    self.respond(handler, 200, self.status(settings, authenticated))
                    return True
                if path == "/session/connection" and handler.command == "GET":
                    self.respond(handler, 200, {"connected": self.api_origin is not None, "url": self.api_origin or ""})
                    return True
                if path == "/session/connection" and handler.command == "PUT":
                    connection = self.json_body(handler, ("url", "key"))
                    self.validate_connection(connection)
                    connection["url"] = connection["url"].rstrip("/")
                    write_json(self.connection_file, connection)
                    self.api_origin = connection["url"]
                    self.api_token = connection["key"]
                    self.respond(handler, 200, {"connected": True, "url": self.api_origin})
                    return True
                if path == "/session/connection" and handler.command == "DELETE":
                    if self.connection_file.exists():
                        self.connection_file.unlink()
                    self.api_origin = None
                    self.api_token = self.local_token
                    self.respond(handler, 200, {"connected": False, "url": ""})
                    return True
                if path == "/session/logout-others" and handler.command == "POST":
                    sessions = self.sessions()
                    current_hash = self.session_hash(handler)
                    kept = {}
                    if current_hash in sessions:
                        kept[current_hash] = sessions[current_hash]
                    write_json(self.sessions_file, kept)
                    self.respond(handler, 200, self.status(settings, authenticated))
                    return True
                if path == "/session/logout" and handler.command == "POST":
                    sessions = self.sessions()
                    sessions.pop(self.session_hash(handler), None)
                    write_json(self.sessions_file, sessions)
                    self.respond(handler, 200, self.status(settings, not settings["enabled"]),
                                 COOKIE + "=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0")
                    return True
                if path.startswith("/session/"):
                    handler.json_error(404, "Unknown portal session action")
                    return True
        except (ValueError, OSError, KeyError, TypeError):
            handler.json_error(400 if path in ("/session/login", "/session/password", "/session/settings", "/session/connection") else 503,
                               "Portal access request could not be applied")
            return True
        return False


def initialize(state_dir: Any, *, public_origin: Any = "http://127.0.0.1:8767", bind: Any = "0.0.0.0", port: Any = 8767) -> Any:
    """Write local installation files. This never installs or starts a service."""
    state_dir = Path(state_dir).resolve()
    state_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    root = state_dir / "workspace"
    key = state_dir / "controller.key"
    credential = state_dir / "api-credential"
    permission = state_dir / "permission.json"
    control.initialize_key(key)
    control.initialize_key(credential)
    previous = os.environ.get("PREDREAM_ADMIN_KEY_FILE")
    os.environ["PREDREAM_ADMIN_KEY_FILE"] = str(key)
    try:
        control.issue_permission(permission, admin_id="dreambox:head-owner", root_scope=root,
                                 modes=["spark"], autonomy=["manual", "supervised", "automatic"])
    finally:
        if previous is None:
            os.environ.pop("PREDREAM_ADMIN_KEY_FILE")
        else:
            os.environ["PREDREAM_ADMIN_KEY_FILE"] = previous
    config = {"root": str(root), "credential_file": str(credential), "key_file": str(key),
              "permission_file": str(permission), "password_file": str(state_dir / "password.json"),
              "mode": "spark", "bind": bind, "port": port, "public_origin": public_origin}
    HostAccess(credential, config["password_file"], public_origin)
    write_json(state_dir / "host-config.json", config)
    unit = files("glm_rsi_dream").joinpath("resources/admin_host/rsi-dreambox.service").read_text()
    unit = unit.replace("@CONFIG@", str(state_dir / "host-config.json"))
    unit = unit.replace("@PYTHON@", os.path.abspath(sys.executable))
    unit = unit.replace("@USER@", pwd.getpwuid(os.getuid()).pw_name)
    (state_dir / "rsi-dreambox.service").write_text(unit)
    return {"config_file": str(state_dir / "host-config.json"), "unit_file": str(state_dir / "rsi-dreambox.service"),
            "portal": public_origin, "password_enabled": True, "setup_required": True, "service_started": False}


def main(argv: Any = None) -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Prepare private files and a service unit, without starting it")
    init.add_argument("--state-dir", type=Path, required=True)
    init.add_argument("--public-origin", default="http://127.0.0.1:8767")
    init.add_argument("--bind", default="0.0.0.0")
    init.add_argument("--port", type=int, default=8767)
    serve = commands.add_parser("serve", help="Run the separate head portal process")
    serve.add_argument("--config", type=Path, required=True)
    serve.add_argument("--enable-dream-features", action="store_true",
                       help="Owner opening after pocket-OFF recall qualification; manual preparation only")
    args = parser.parse_args(argv)
    if args.command == "init":
        print(json.dumps(initialize(args.state_dir, public_origin=args.public_origin, bind=args.bind, port=args.port)))
        return
    config = read_json(args.config)
    os.environ["PREDREAM_MODE"] = config["mode"]
    os.environ["PREDREAM_AUTONOMY"] = "manual"
    os.environ["PREDREAM_ADMIN_KEY_FILE"] = config["key_file"]
    os.environ["PREDREAM_ADMIN_PERMISSION_FILE"] = config["permission_file"]
    access = HostAccess(config["credential_file"], config["password_file"], config["public_origin"])
    api = create_api(config["root"], config["credential_file"], port=0,
                     dream_features_enabled=args.enable_dream_features)
    try:
        portal = create_portal(config["port"], "http://127.0.0.1:" + str(api.server_port),
                               access=access, bind=config["bind"])
        api.allowed_origins.add(access.origin)
        thread = threading.Thread(target=api.serve_forever, kwargs={"poll_interval": 0.05})
        thread.start()
        print(json.dumps({"portal": access.origin, "mode": config["mode"], "hosted": True}), flush=True)
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
