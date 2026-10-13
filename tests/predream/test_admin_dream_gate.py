"""The server-owned dream gate is closed by default and has an explicit opening path."""

import json
import threading
from contextlib import contextmanager
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from glm_rsi_dream.predream.admin_api import create_server, read_token
from glm_rsi_dream.predream.admin_demo import demo_authority, prepare_fixture
from glm_rsi_dream.predream.admin_schema import OPERATIONS, openapi_spec

DREAM_OPERATIONS = []
for method, path, operation, schema, _ in OPERATIONS:
    if operation.startswith(("dreams_", "schedule_")):
        DREAM_OPERATIONS.append((method.upper(), path, schema))


@contextmanager
def server_for(root, token_file, **settings):
    server = create_server(root, token_file, port=0, **settings)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_port)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def call(url, token, method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}
    request = Request(url + "/api/v1" + path, data=data, headers=headers, method=method)
    try:
        response = urlopen(request, timeout=5)
    except HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response)


@pytest.fixture(params=["spark", "harness"])
def closed_api(tmp_path, request):
    root = tmp_path / "closed"
    with demo_authority(root, request.param) as authority:
        receipt = prepare_fixture(root)
        assert receipt["dream_features_enabled"] is False
        with server_for(root, authority["token_file"]) as url:
            yield root, authority, url, read_token(authority["token_file"])


@pytest.mark.parametrize("method,path,schema", DREAM_OPERATIONS)
def test_every_dream_endpoint_rejects_before_body_validation_or_storage(closed_api, method, path, schema):
    root, _, url, token = closed_api
    state_before = (root / "admin/state.json").read_bytes()
    queue_before = (root / "queue/candidates.jsonl").read_bytes()
    # An absent or malformed operation body still gets the same closed feature response.
    status, result = call(url, token, method, path.replace("{id}", "fixture-dream"), {} if schema else None)
    assert status == 503 and result == {"error": "feature_unavailable: under development"}
    assert (root / "admin/state.json").read_bytes() == state_before
    assert (root / "queue/candidates.jsonl").read_bytes() == queue_before


def test_closed_capability_core_actions_and_client_flag_tampering(closed_api):
    _, _, url, token = closed_api
    status, system = call(url, token, "GET", "/system")
    assert status == 200
    assert system["features"]["dreams"] == {"enabled": False, "label": "Under development",
        "status": "Under development", "owner": "dream-neurons",
        "opening_criterion": "Pocket-OFF recall bar passes"}
    assert call(url, "wrong-credential", "GET", "/dreams")[0] == 401
    for path in ("/rules", "/queue", "/pockets", "/control", "/policy"):
        assert call(url, token, "GET", path)[0] == 200
    assert call(url, token, "POST", "/pockets", {"id": "usable", "topic": "aster", "capacity": 2})[0] == 201
    assert call(url, token, "PUT", "/schedule", {"dream_features_enabled": True})[0] == 503
    current = call(url, token, "GET", "/control")[1]
    body = {"expected_revision": current["revision"], "autonomy": "manual", "auto_routes": [],
            "min_score": 0.9, "dream_features_enabled": True}
    assert call(url, token, "PUT", "/control", body)[0] == 400
    assert call(url, token, "GET", "/system")[1]["features"]["dreams"]["enabled"] is False
    spec = openapi_spec()
    for method, path, _ in DREAM_OPERATIONS:
        assert spec["paths"]["/api/v1" + path][method.lower()]["x-feature-gate"] == "dream_features_enabled"


@pytest.mark.parametrize("mode", ["spark", "harness"])
def test_explicit_server_opening_retains_bearer_grant_and_manual_boundary(tmp_path, mode):
    root = tmp_path / "opened-fixture"
    with demo_authority(root, mode) as authority:
        prepare_fixture(root, dream_features_enabled=True)
        with server_for(root, authority["token_file"], dream_features_enabled=True) as url:
            token = read_token(authority["token_file"])
            feature = call(url, token, "GET", "/system")[1]["features"]["dreams"]
            assert feature["enabled"] and feature["status"] == "Manual preparation"
            assert call(url, "wrong", "GET", "/schedule")[0] == 401
            status, result = call(url, token, "POST", "/dreams/fixture-dream/check",
                                 {"at": "2026-10-13T21:00:00-04:00", "idle_seconds": 900, "active_requests": 0})
            assert status == 200 and result["eligible"] and result["executes_training"] is False
            permission = authority["permission_file"]
            signed = json.loads(permission.read_text())
            signed["permission"]["admin_id"] = "tampered"
            permission.write_text(json.dumps(signed))
            status, result = call(url, token, "DELETE", "/dreams/fixture-dream", {"expected_revision": 1})
            assert status == 403 and "grant denied" in result["error"]


@pytest.mark.parametrize("value", [1, "true", None])
def test_feature_flag_accepts_only_a_host_boolean(tmp_path, value):
    with demo_authority(tmp_path / "root", "harness") as authority:
        with pytest.raises(ValueError, match="boolean"):
            create_server(tmp_path / "root", authority["token_file"], port=0, dream_features_enabled=value)
