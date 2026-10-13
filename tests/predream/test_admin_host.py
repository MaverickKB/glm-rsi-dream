"""Actual hosted portal sessions, protected API and first-run installation."""

import hashlib
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from glm_rsi_dream.predream import admin_host
from glm_rsi_dream.predream.admin_api import create_server as create_api
from glm_rsi_dream.predream.admin_demo import demo_authority, prepare_fixture
from glm_rsi_dream.predream.admin_host import COOKIE, HostAccess, initialize
from glm_rsi_dream.predream.admin_portal import create_server as create_portal


class HeadPortal:
    def __init__(self, url, access, root):
        self.url = url
        self.access = access
        self.root = root
        self.cookie = ""

    def call(self, path, method="GET", body=None, *, cookie=None, origin=None, headers=None):
        request_headers = {"Content-Type": "application/json", "Origin": self.url if origin is None else origin}
        value = self.cookie if cookie is None else cookie
        if value:
            request_headers["Cookie"] = value
        request_headers.update(headers or {})
        data = json.dumps(body).encode() if body is not None else None
        request = Request(self.url + path, data=data, headers=request_headers, method=method)
        try:
            response = urlopen(request, timeout=5)
        except HTTPError as error:
            response = error
        with response:
            raw = response.read()
            content_type = response.headers.get_content_type()
            if content_type == "application/json":
                value = json.loads(raw)
            elif content_type == "image/png":
                value = raw
            else:
                value = raw.decode()
            cookie_header = response.headers.get("Set-Cookie")
            if cookie_header:
                self.cookie = cookie_header.split(";")[0]
            return response.status, value, dict(response.headers)

    def ok(self, path, method="GET", body=None):
        status, value, headers = self.call(path, method, body)
        assert status == 200, value
        return value, headers

    def setup(self, password="fictional portal password"):
        return self.ok("/session/password", "PUT", {"password": password})


@contextmanager
def hosted(root, authority):
    api = create_api(root, authority["token_file"], port=0)
    access = HostAccess(authority["token_file"], root / "host/password.json", "http://127.0.0.1:0")
    portal = create_portal(0, "http://127.0.0.1:" + str(api.server_port), access=access)
    api.allowed_origins.add(access.origin)
    threads = []
    for server in (api, portal):
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        threads.append(thread)
    try:
        yield HeadPortal(access.origin, access, root)
    finally:
        for server, thread in zip((api, portal), threads, strict=True):
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()


@pytest.fixture(params=["spark", "harness"])
def head(tmp_path, request):
    root = tmp_path / "workspace"
    with demo_authority(root, request.param) as authority:
        prepare_fixture(root)
        with hosted(root, authority) as portal:
            yield portal


def test_default_locked_first_run_and_server_loaded_credential(head):
    status, _ = head.ok("/session/status")
    assert status == {"hosted": True, "authenticated": False, "password_enabled": True,
                      "setup_required": True, "session_seconds": 86400}
    for path in ("/", "/index.html", "/help.json", "/demo.json", "/openapi.json", "/api/v1/system"):
        assert head.call(path)[0] == 401
    for path in ("/favicon.png", "/hero.png"):
        image_status, image, image_headers = head.call(path)
        assert image_status == 200
        assert image_headers["Content-Type"] == "image/png"
        assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert head.call("/session/password", "PUT", {"password": ""})[0] == 400
    value, headers = head.setup()
    assert value["authenticated"] and not value["setup_required"]
    assert "HttpOnly" in headers["Set-Cookie"] and "SameSite=Strict" in headers["Set-Cookie"]
    assert "Max-Age=86400" in headers["Set-Cookie"]
    system, _ = head.ok("/api/v1/system")
    assert system["control"]["mode"] in ("spark", "harness")
    assert not system["features"]["dreams"]["enabled"]
    assert head.access.api_token not in json.dumps(system) + json.dumps(value) + json.dumps(headers)
    settings = json.loads(head.access.password_file.read_text())
    assert settings["hash"] != "fictional portal password"
    assert head.access.password_file.stat().st_mode & 0o077 == 0
    sessions = json.loads(head.access.sessions_file.read_text())
    token = head.cookie.split("=", 1)[1]
    assert token not in head.access.sessions_file.read_text()
    assert hashlib.sha256(token.encode()).hexdigest() in sessions
    assert head.access.sessions_file.stat().st_mode & 0o077 == 0
    for level in ("supervised", "automatic", "manual"):
        current, _ = head.ok("/api/v1/control")
        selected, _ = head.ok("/api/v1/control", "PUT", {"expected_revision": current["revision"],
            "autonomy": level, "auto_routes": ["memory"], "min_score": 0.9})
        assert selected["autonomy"] == level


def test_absolute_expiry_is_never_extended_by_activity(head, monkeypatch):
    clock = [2000000000.0]
    monkeypatch.setattr(admin_host.time, "time", lambda: clock[0])
    head.setup()
    before = head.access.sessions_file.read_bytes()
    for offset in (60, 3600, 86399):
        clock[0] = 2000000000.0 + offset
        head.ok("/api/v1/queue")
        assert head.access.sessions_file.read_bytes() == before
    clock[0] = 2000000000.0 + 86400
    assert head.call("/api/v1/queue")[0] == 401
    assert head.ok("/session/status")[0]["authenticated"] is False


def test_logout_and_password_change_invalidate_real_sessions(head):
    head.setup()
    first = head.cookie
    head.ok("/session/login", "POST", {"password": "fictional portal password"})
    second = head.cookie
    head.ok("/session/logout-others", "POST", {})
    assert head.call("/api/v1/system", cookie=first)[0] == 401
    head.cookie = second
    head.ok("/session/password", "PUT", {"password": "changed fictional password"})
    changed = head.cookie
    assert head.call("/api/v1/system", cookie=second)[0] == 401
    head.cookie = changed
    head.ok("/session/logout", "POST", {})
    assert head.call("/api/v1/system", cookie=changed)[0] == 401
    assert head.call("/session/login", "POST", {"password": "fictional portal password"})[0] == 401
    head.ok("/session/login", "POST", {"password": "changed fictional password"})


def test_explicit_unlock_and_relock_keep_server_enforcement(head):
    head.setup()
    head.ok("/session/password", "PUT", {"password": ""})
    assert head.ok("/session/status")[0]["password_enabled"] is False
    assert head.call("/api/v1/system", cookie="")[0] == 200
    head.setup("relocked fictional password")
    assert head.call("/api/v1/system", cookie="")[0] == 401


def test_spoofed_cookie_csrf_host_and_login_rate_limit(head):
    head.setup()
    for value in (COOKIE + "=" + "a" * 64, COOKIE + "=bad"):
        assert head.call("/api/v1/system", cookie=value)[0] == 401
    assert head.call("/session/password", "PUT", {"password": ""}, origin="http://foreign.invalid")[0] == 403
    assert head.call("/session/settings", "PUT", {"session_seconds": 900}, origin="")[0] == 403
    assert head.call("/api/v1/control", "PUT", {}, origin="http://foreign.invalid")[0] == 403
    assert head.call("/api/v1/system", headers={"Host": "foreign.invalid"})[0] == 421
    for _ in range(5):
        assert head.call("/session/login", "POST", {"password": "wrong"}, cookie="")[0] == 401
    assert head.call("/session/login", "POST", {"password": "fictional portal password"}, cookie="")[0] == 429


@pytest.mark.parametrize("seconds", [299, 86401, True, "3600", 1.5])
def test_session_lifetime_range_is_server_validated(head, seconds):
    head.setup()
    previous = head.access.password_file.read_bytes()
    assert head.call("/session/settings", "PUT", {"session_seconds": seconds})[0] == 400
    assert head.access.password_file.read_bytes() == previous


def test_timeout_presets_and_custom_apply_only_to_new_logins(head):
    head.setup()
    previous = head.access.sessions_file.read_bytes()
    for seconds in (900, 3600, 28800, 86400, 300, 1800):
        settings, _ = head.ok("/session/settings", "PUT", {"session_seconds": seconds})
        assert settings["session_seconds"] == seconds
        assert head.access.sessions_file.read_bytes() == previous
    head.ok("/session/login", "POST", {"password": "fictional portal password"})
    current_hash = hashlib.sha256(head.cookie.split("=", 1)[1].encode()).hexdigest()
    session = json.loads(head.access.sessions_file.read_text())[current_hash]
    assert session["expires_at"] - session["login_at"] == 1800


def test_https_cookie_and_private_session_file_enforcement(head):
    head.setup()
    head.access.password_settings()
    settings = head.access.password_settings()
    head.access.origin = head.access.origin.replace("http://", "https://")
    assert "; Secure" in head.access.cookie("a" * 64, settings)
    sessions = head.access.sessions_file.read_bytes()
    head.access.sessions_file.chmod(0o644)
    assert head.call("/api/v1/system", headers={"Host": head.access.origin.split("//")[1]}, origin=head.access.origin)[0] == 503
    head.access.sessions_file.chmod(0o600)
    assert head.access.sessions_file.read_bytes() == sessions


def test_session_survives_a_host_restart(tmp_path):
    root = tmp_path / "restart"
    with demo_authority(root, "spark") as authority:
        prepare_fixture(root)
        with hosted(root, authority) as first:
            first.setup()
            cookie = first.cookie
            original = first.access.sessions_file.read_bytes()
        with hosted(root, authority) as restarted:
            assert restarted.call("/api/v1/system", cookie=cookie)[0] == 200
            assert restarted.access.sessions_file.read_bytes() == original


def test_advanced_engine_connection_stays_on_the_host_and_resets(head):
    head.setup()
    remote_token = head.root / "remote.token"
    remote_token.write_text("remote_fixture_token_" + "r" * 32)
    remote_token.chmod(0o600)
    remote = create_api(head.root, remote_token, port=0)
    thread = threading.Thread(target=remote.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    remote_url = "http://127.0.0.1:" + str(remote.server_port)
    try:
        connected, _ = head.ok("/session/connection", "PUT", {"url": remote_url, "key": remote_token.read_text()})
        assert connected == {"connected": True, "url": remote_url}
        assert "key" not in connected
        assert head.access.connection_file.stat().st_mode & 0o077 == 0
        assert head.ok("/session/connection")[0] == connected
        head.ok("/api/v1/system")
        head.ok("/openapi.json")
        assert head.call("/session/connection", "PUT", {"url": remote_url, "key": remote_token.read_text()}, cookie="")[0] == 401
        assert head.call("/session/connection", "PUT", {"url": remote_url, "key": remote_token.read_text()}, origin="http://foreign.invalid")[0] == 403
        assert head.call("/session/connection", "PUT", {"url": remote_url + "/private", "key": remote_token.read_text()})[0] == 400
        head.ok("/session/connection", "DELETE")
        assert not head.access.connection_file.exists()
        assert head.ok("/session/connection")[0] == {"connected": False, "url": ""}
        head.ok("/api/v1/system")
    finally:
        remote.shutdown()
        thread.join(timeout=5)
        remote.server_close()


def test_install_files_are_head_default_private_and_never_start_service(tmp_path):
    result = initialize(tmp_path / "installation")
    assert result["portal"] == "http://127.0.0.1:8767"
    assert result["password_enabled"] and result["setup_required"]
    assert result["service_started"] is False
    config = json.loads(Path(result["config_file"]).read_text())
    assert config["mode"] == "spark" and config["port"] == 8767
    for field in ("credential_file", "key_file", "permission_file"):
        assert Path(config[field]).stat().st_mode & 0o077 == 0
    unit = Path(result["unit_file"]).read_text()
    assert "admin_host serve" in unit and "@" not in unit
    assert config["credential_file"] not in unit
    assert "8888" not in unit
    with pytest.raises(FileExistsError):
        initialize(tmp_path / "installation")
