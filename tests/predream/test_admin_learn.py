"""Real HTTP prepared intake, GLM proposals, signed decisions and retry behavior."""

import copy
import json
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from glm_rsi_dream.predream import control
from glm_rsi_dream.predream.absorb import absorb, stage
from glm_rsi_dream.predream.admin_api import create_server
from glm_rsi_dream.predream.manual import import_learned
from glm_rsi_dream.predream.policy import record_policy
from glm_rsi_dream.predream.queue import append_record, latest_records


class ScriptedGLM:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.messages = []
        self.on_request = None

    def chat(self, messages):
        self.messages.append(messages)
        if self.on_request is not None:
            self.on_request()
            self.on_request = None
        if not self.answers:
            raise AssertionError("Unexpected model request in CPU fixture")
        answer = self.answers.pop(0)
        return answer if isinstance(answer, str) else json.dumps(answer)


def review(kind="new", targets=None, safe=True):
    return {"safe": safe, "kind": kind, "targets": targets or [], "reason": "Fictional fixture review"}


def decision(route="pocket", verdict="accept"):
    return {"verdict": verdict, "route": route, "pocket": "lumen" if route == "pocket" else None,
            "reason": "Fictional fixture destination", "score": 0.95}


def pack(text="Fictional Lumen has three sample bays.", evidence=None):
    fact = {"statement": text}
    if evidence is not None:
        fact["prompt_evidence"] = evidence
    return {"source_ref": "fixture:prepared", "format": "json", "content": json.dumps({"facts": [fact]})}


@pytest.fixture(params=["spark", "harness"])
def api(tmp_path, monkeypatch, request):
    root = tmp_path / "workspace"
    key = tmp_path / "signing.key"
    token = tmp_path / "http.token"
    permission = tmp_path / "permission.json"
    control.initialize_key(key)
    control.initialize_key(token)
    monkeypatch.setenv("PREDREAM_ADMIN_KEY_FILE", str(key))
    monkeypatch.setenv("PREDREAM_ADMIN_PERMISSION_FILE", str(permission))
    monkeypatch.setenv("PREDREAM_MODE", request.param)
    monkeypatch.setenv("PREDREAM_AUTONOMY", "manual")
    control.issue_permission(permission, admin_id="fixture:prepared-admin", root_scope=root,
                             modes=[request.param], autonomy=["manual", "supervised", "automatic"])
    client = ScriptedGLM()
    server = create_server(root, token, port=0, client=client)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()

    def call(method, path, body=None, auth=True):
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = "Bearer " + token.read_text().strip()
        data = json.dumps(body).encode() if body is not None else None
        req = Request("http://127.0.0.1:" + str(server.server_port) + "/api/v1" + path,
                      data=data, headers=headers, method=method)
        try:
            response = urlopen(req, timeout=5)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.load(response)

    yield SimpleNamespace(root=root, permission=permission, client=client, call=call)
    server.shutdown()
    thread.join(timeout=5)
    server.server_close()


def grade_item(api, item):
    status, result = api.call("POST", "/queue/" + item["id"] + "/grade",
                              {"expected_revision": item["revision"]})
    assert status == 200, result
    return result


@pytest.mark.parametrize("route", ["weights", "pocket", "memory"])
def test_pack_review_approval_and_handoff_use_the_normal_queue(api, route):
    status, submitted = api.call("POST", "/learn", pack())
    assert status == 200 and len(submitted["queued_ids"]) == 1
    assert not api.client.messages
    item = submitted["items"][0]
    assert item["source"]["kind"] == "manual" and item["novelty"] == "unmeasured"
    assert item["timings_ms"]["probe"] == 0 and item["status"] == "pending"
    api.client.answers = [review(), decision(route)]
    result = grade_item(api, item)
    proposed = result["item"]
    assert not result["errors"] and proposed["status"] == "pending"
    assert proposed["proposal"]["grade"]["route"] == route
    assert len(api.client.messages) == 2
    status, approved = api.call("POST", "/queue/" + item["id"] + "/approve",
                               {"expected_revision": proposed["revision"]})
    assert status == 200 and approved["status"] == "accepted"
    if route == "memory":
        assert absorb(api.root)["absorbed"] == [item["id"]]
    else:
        assert stage(api.root, "prepared")["items"][0]["id"] == item["id"]
    assert not list(api.root.glob("training*"))


def test_text_extraction_then_review_has_no_blind_probe_and_replays_exactly(api):
    text = "Fictional Lumen has three sample bays."
    api.client.answers = [{"candidates": [{"text": text, "prompt_evidence": text, "probe_question": "How many bays?"}]}]
    body = {"source_ref": "fixture:note", "format": "text", "content": text}
    status, first = api.call("POST", "/learn", body)
    assert status == 200
    api.client.answers = [review(), decision()]
    item = grade_item(api, first["items"][0])["item"]
    before = (api.root / "queue/candidates.jsonl").read_bytes()
    status, replay = api.call("POST", "/learn", body)
    assert status == 200 and replay["queued_ids"] == [] and replay["items"] == [item]
    assert grade_item(api, item)["item"] == item
    assert len(api.client.messages) == 3
    assert (api.root / "queue/candidates.jsonl").read_bytes() == before
    changed = dict(body, content=text + " Revised specification.")
    api.client.answers = [{"candidates": [{"text": text, "prompt_evidence": text, "probe_question": "How many bays?"}]}]
    status, new = api.call("POST", "/learn", changed)
    assert status == 200 and new["items"][0]["id"] != item["id"]
    assert len(latest_records(api.root / "queue/candidates.jsonl")) == 2


@pytest.mark.parametrize("content", ["{}", '{"facts":[]}', '{"facts":[{"statement":" "}]}',
    '{"facts":[{"statement":"Fact","route":"weights"}]}', '{"facts":[{"statement":"Fact"}],"command":"sh"}'])
def test_invalid_pack_is_rejected_before_storage_or_model_access(api, content):
    status, _ = api.call("POST", "/learn", dict(pack(), content=content))
    assert status == 400 and not api.client.messages
    assert not (api.root / "queue/candidates.jsonl").exists()
    assert not list(api.root.glob("intake/manual/*.json"))


@pytest.mark.parametrize("encoded", [False, True])
def test_credential_preflight_covers_decoded_pack_fields_and_stores_nothing(api, encoded):
    secret = "api_key=" + "sk-" + "x" * 24
    body = pack(secret)
    if encoded:
        escaped = ""
        for character in secret:
            escaped += "\\u" + format(ord(character), "04x")
        body["content"] = '{"facts":[{"statement":"' + escaped + '"}]}'
    status, _ = api.call("POST", "/learn", body)
    assert status == 400 and not api.client.messages
    assert not (api.root / "queue/candidates.jsonl").exists()
    assert not list(api.root.rglob("*.json"))


def test_bearer_and_scoped_grant_gate_prepared_intake(api):
    assert api.call("POST", "/learn", pack(), auth=False)[0] == 401
    api.permission.unlink()
    assert api.call("POST", "/learn", pack())[0] == 403
    assert not api.client.messages and not (api.root / "queue/candidates.jsonl").exists()


def test_failed_review_stays_pending_and_can_retry_without_duplicate_intake(api):
    _, submitted = api.call("POST", "/learn", pack())
    api.client.answers = ["invalid", "invalid", "invalid"]
    failed = grade_item(api, submitted["items"][0])
    assert failed["errors"] and failed["item"]["status"] == "pending"
    assert "proposal" not in failed["item"]
    _, replay = api.call("POST", "/learn", pack())
    assert replay["queued_ids"] == []
    api.client.answers = [review(), decision()]
    assert grade_item(api, replay["items"][0])["item"]["proposal"]["grade"]["verdict"] == "accept"
    assert len(latest_records(api.root / "queue/candidates.jsonl")) == 1


@pytest.mark.parametrize("kind", ["duplicate", "update"])
def test_learned_relationship_keeps_the_old_receipt_and_skips_novelty_rejection(api, kind):
    path = api.root.parent / "learned.json"
    path.write_text(json.dumps({"facts": [{"id": "lumen-bays", "statement": "Fictional Lumen has two sample bays.",
                                         "receipt": "fixture:prior-learning"}]}))
    old = import_learned(api.root, path)[0]
    text = old["text"] if kind == "duplicate" else "Fictional Lumen now has three sample bays."
    _, submitted = api.call("POST", "/learn", pack(text))
    api.client.answers = [review(kind, [old["id"]]), decision()]
    item = grade_item(api, submitted["items"][0])["item"]
    assert item["change"]["kind"] == kind and item["change"]["targets"][0]["receipt"] == "fixture:prior-learning"
    assert item["proposal"]["grade"]["verdict"] == "accept"
    assert latest_records(api.root / "queue/candidates.jsonl")[0] == old


@pytest.mark.parametrize("autonomy", ["supervised", "automatic"])
def test_prepared_path_obeys_permission_bounded_auto_decisions(api, autonomy):
    status, _ = api.call("PUT", "/control", {"expected_revision": 1, "autonomy": autonomy,
                                            "auto_routes": ["memory"], "min_score": 0.9})
    assert status == 200
    _, submitted = api.call("POST", "/learn", pack("Fictional Mira departs tonight."))
    api.client.answers = [review(), decision("memory")]
    item = grade_item(api, submitted["items"][0])["item"]
    assert item["status"] == "accepted" and item["approval"]["signature"]


def test_active_custom_policy_still_requires_safe_model_review(api):
    record_policy(api.root, {"schema_version": 1, "name": "fixture:unsafe", "extract": {},
                            "grade": {"strategy": "custom", "instructions": "Custom fixture rubric"}})
    _, submitted = api.call("POST", "/learn", pack())
    api.client.answers = [review(safe=False), decision()]
    result = grade_item(api, submitted["items"][0])
    assert result["errors"] and result["item"]["status"] == "pending"
    assert "Custom fixture rubric" in api.client.messages[1][0]["content"]


def test_concurrent_queue_edit_survives_in_flight_grade(api):
    _, submitted = api.call("POST", "/learn", pack())
    item = submitted["items"][0]

    def hold_during_review():
        held = copy.deepcopy(latest_records(api.root / "queue/candidates.jsonl")[0])
        held["admin_review"] = {"hold": True, "note": "Concurrent review"}
        append_record(api.root / "queue/candidates.jsonl", held)

    api.client.on_request = hold_during_review
    api.client.answers = [review(), decision()]
    status, _ = api.call("POST", "/queue/" + item["id"] + "/grade", {"expected_revision": item["revision"]})
    assert status == 409
    current = latest_records(api.root / "queue/candidates.jsonl")[0]
    assert current["admin_review"]["hold"] and "proposal" not in current


@pytest.mark.parametrize("method,path,body", [
    ("GET", "/schedule", None), ("PUT", "/schedule", {}),
    ("GET", "/dreams", None), ("POST", "/dreams", {}),
    ("GET", "/dreams/fixture", None), ("DELETE", "/dreams/fixture", {}),
    ("POST", "/dreams/fixture/check", {}),
])
def test_default_dream_flag_denies_every_api_surface_before_dispatch(api, method, path, body):
    status, result = api.call(method, path, body)
    assert status == 503 and result == {"error": "feature_unavailable: under development"}
    assert not api.root.exists() and not api.client.messages


def test_dream_status_is_server_owned_and_normal_features_remain_usable(api):
    status, system = api.call("GET", "/system")
    assert status == 200 and system["features"]["dreams"]["enabled"] is False
    assert system["features"]["dreams"]["label"] == "Under development"
    for path in ("/queue", "/pockets", "/rules", "/policy", "/control"):
        assert api.call("GET", path)[0] == 200
    assert api.call("POST", "/learn", dict(pack(), dream_features_enabled=True))[0] == 400
    assert api.call("GET", "/dreams")[0] == 503


@pytest.mark.parametrize("method,path", [("POST", "/dreams"), ("PUT", "/schedule"), ("DELETE", "/dreams/fixture")])
def test_closed_gate_precedes_json_body_parsing(api, method, path):
    status, result = api.call(method, path)
    assert status == 503 and result["error"] == "feature_unavailable: under development"
    assert not api.root.exists()
