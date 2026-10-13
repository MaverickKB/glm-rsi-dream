"""The shipped JavaScript client, preview and real API agree on the wire contract."""

import json
import os
import re
import shutil
import subprocess
import threading
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from glm_rsi_dream.predream import control
from glm_rsi_dream.predream.absorb import absorb
from glm_rsi_dream.predream.admin_api import create_server as create_api
from glm_rsi_dream.predream.admin_api import read_token
from glm_rsi_dream.predream.admin_demo import (
    FixtureGLM,
    demo_authority,
    prepare_fixture,
)
from glm_rsi_dream.predream.admin_portal import create_server as create_portal
from glm_rsi_dream.predream.admin_schema import openapi_spec
from glm_rsi_dream.predream.queue import latest_records

CLIENT = Path(__file__).parent / "fixtures/admin_contract_client.cjs"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Portal client contracts require Node 18 or later")


def input_field(value, kind="text", *, json_value=False, checked=False):
    return {"value": str(value), "type": kind, "checked": checked,
            "dataset": {"json": "true"} if json_value else {}}


class Portal:
    def __init__(self, url, authority, root, fixture):
        self.url = url
        self.authority = authority
        self.root = root
        self.fixture = fixture
        self.token = read_token(authority["token_file"])
        self.spec = openapi_spec()
        resource = Resource.from_contents(self.spec, default_specification=DRAFT202012)
        self.registry = Registry().with_resource("urn:rsi-admin", resource)

    def call(self, path, method="GET", body=None, *, status=200, token=None,
             schema=None, fields=None, previous=None, representation=None):
        payload = {"script": str(files("glm_rsi_dream").joinpath("resources/admin_portal/portal.js")),
                   "origin": self.url, "token": self.token if token is None else token,
                   "spec": self.spec, "path": path, "method": method, "body": body,
                   "schemaName": schema, "fields": fields or {}, "previous": previous or {},
                   "representation": representation or {}}
        process = subprocess.run([NODE, str(CLIENT)], input=json.dumps(payload),
                                 capture_output=True, text=True, timeout=15, check=True)
        result = json.loads(process.stdout)
        assert result["ok"] == (status < 400), result
        if status >= 400:
            assert result["status"] == status, result
        operation = None
        for route, methods in self.spec["paths"].items():
            pattern = route.replace("{id}", "[a-z0-9][a-z0-9_-]{0,63}")
            if re.fullmatch(pattern, "/api/v1" + path) and method.lower() in methods:
                operation = methods[method.lower()]
                break
        assert operation is not None, path
        response_schema = operation["responses"][str(status)]["content"]["application/json"]["schema"]
        assert result["requests"][-1]["status"] == status, result
        response = result["requests"][-1]["response"]
        self.validate(response, response_schema)
        if status < 400 and "requestBody" in operation:
            request_schema = operation["requestBody"]["content"]["application/json"]["schema"]
            self.validate(result["body"], request_schema)
        receipt_path = os.environ.get("RSI_ADMIN_CONTRACT_RECEIPTS")
        if receipt_path:
            receipt = {"mode": os.environ["PREDREAM_MODE"], "operation": operation["operationId"],
                       "status": status, "request": result["requests"][-1],
                       "schema_validated": True, "client_ok": result["ok"]}
            with Path(receipt_path).open("a", encoding="utf-8") as file:
                file.write(json.dumps(receipt, ensure_ascii=False) + "\n")
        return result

    def validate(self, value, schema):
        document = {"$ref": "urn:rsi-admin" + schema["$ref"]}
        # Local references resolve against the complete packaged OpenAPI document.
        validator = Draft202012Validator(document, registry=self.registry, format_checker=FormatChecker())
        validator.validate(value)

    def get(self, path):
        return self.call(path)["result"]

    def approve(self, identifier):
        row = self.get("/queue/" + identifier)
        return self.call("/queue/" + identifier + "/approve", "POST",
                         {"expected_revision": row["revision"]})["result"]


@contextmanager
def running_portal(root, authority, fixture, *, dream_features_enabled=False):
    api = create_api(root, authority["token_file"], port=0, client=FixtureGLM(root),
                     dream_features_enabled=dream_features_enabled)
    api_url = "http://127.0.0.1:" + str(api.server_port)
    preview = create_portal(api_url=api_url)
    url = "http://127.0.0.1:" + str(preview.server_port)
    api.allowed_origins.add(url)
    threads = []
    for server in (api, preview):
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        threads.append(thread)
    try:
        yield Portal(url, authority, root, fixture)
    finally:
        for server, thread in zip((api, preview), threads):
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()


@pytest.fixture(params=["spark", "harness"])
def portal(tmp_path, request):
    root = tmp_path / "workspace"
    with demo_authority(root, request.param) as authority:
        fixture = prepare_fixture(root, dream_features_enabled=True)
        with running_portal(root, authority, fixture, dream_features_enabled=True) as instance:
            yield instance


def test_all_six_pages_read_real_typed_collections(portal):
    packaged = json.loads(files("glm_rsi_dream").joinpath("resources/rsi-admin-openapi.json").read_text())
    assert packaged == portal.spec
    assert packaged["info"]["title"] == "RSI Dreambox API"
    for path in ("/system", "/control", "/rules", "/queue", "/schedule", "/dreams", "/pockets", "/policy"):
        portal.get(path)
    assert portal.get("/system")["control"]["mode"] in ("spark", "harness")
    assert len(portal.get("/queue")["items"]) == 4
    assert portal.get("/system")["collection_enabled"] is False


def test_portal_client_preserves_gate_failures_and_denied_draft(portal):
    denied = portal.call("/queue", token="", status=401)
    assert denied["error"] == "Sign in to continue."
    permission = portal.authority["permission_file"]
    signed = json.loads(permission.read_text())
    signed["permission"]["admin_id"] = "tampered"
    permission.write_text(json.dumps(signed))
    fields = {"id": input_field("denied"), "topic": input_field("aster"),
              "capacity": input_field(2, "number")}
    draft = portal.call("/pockets", "POST", status=403, schema="PocketCreate", fields=fields)
    assert draft["body"] == {"id": "denied", "topic": "aster", "capacity": 2}
    assert portal.get("/pockets")["items"][0]["id"] == "aster"


def test_prepared_pack_and_text_enter_same_queue_and_grade_as_manual_proposals(portal):
    for input_format, text in (("json", "Fictional Aster station's hatch is cobalt."),
                               ("text", "Fictional Iona has a meeting Friday evening.")):
        content = json.dumps({"facts": [{"statement": text}]}) if input_format == "json" else text
        learned = portal.call("/learn", "POST", {"source_ref": "fictional:wire-" + input_format,
            "format": input_format, "content": content})["result"]
        assert learned["novelty_check"] == "skipped_manual"
        assert learned["secret_preflight"] == "clear"
        item = learned["items"][0]
        assert item["source"]["kind"] == "manual" and item["status"] == "pending"
        graded = portal.call("/queue/" + item["id"] + "/grade", "POST",
                             {"expected_revision": item["revision"]})["result"]
        assert not graded["errors"] and graded["item"]["status"] == "pending"
        assert graded["item"]["proposal"]["grade"]["verdict"] == "accept"
        replay = portal.call("/queue/" + item["id"] + "/grade", "POST",
                            {"expected_revision": graded["item"]["revision"]})["result"]
        assert replay == graded
        assert portal.approve(item["id"])["status"] == "accepted"


def test_schema_built_forms_preserve_types_and_stale_revision(portal):
    fields = {"id": input_field("fixture-rule"), "name": input_field("Fixture rule"),
              "domain": input_field("general"), "action": input_field("propose"),
              "route": input_field("memory"), "pocket": input_field(""),
              "priority": input_field(8, "number"), "enabled": input_field("", "checkbox", checked=True),
              "exclude_user_ids": input_field('["excluded-fixture"]', json_value=True)}
    created = portal.call("/rules", "POST", status=201, schema="RuleCreate", fields=fields)
    assert created["result"]["pocket"] is None
    assert created["result"]["priority"] == 8
    assert created["result"]["exclude_user_ids"] == ["excluded-fixture"]
    edited = portal.call("/rules/fixture-rule", "PATCH", schema="RuleEdit",
                         previous={"expected_revision": 1}, fields={"enabled": input_field("", "checkbox")})
    assert edited["result"]["enabled"] is False
    stale = portal.call("/rules/fixture-rule", "PATCH", status=409, schema="RuleEdit",
                        previous={"expected_revision": 1}, fields={"notes": input_field("Preserve this draft")})
    assert stale["body"]["notes"] == "Preserve this draft"
    assert portal.get("/rules/fixture-rule")["revision"] == 2


def test_real_manual_queue_approval_veto_and_memory_handoff(portal):
    ids = portal.fixture["candidate_ids"]
    row = portal.get("/queue/" + ids["pocket"])
    assert row["source"]["kind"] == "manual" and row["novelty"] == "unmeasured"
    held = portal.call("/queue/" + row["id"], "PATCH", schema="QueueEdit",
                       previous={"expected_revision": row["revision"]},
                       fields={"hold": input_field("", "checkbox", checked=True)})["result"]["item"]
    portal.call("/queue/" + row["id"] + "/approve", "POST", {"expected_revision": held["revision"]}, status=400)
    portal.call("/queue/" + row["id"], "PATCH", {"expected_revision": held["revision"], "hold": False})
    accepted = portal.approve(row["id"])
    assert accepted["grade"]["verdict"] == "accept" and accepted["approval"]["decision"]["explicit"]
    veto_row = portal.get("/queue/" + ids["veto"])
    rejected = portal.call("/queue/" + ids["veto"] + "/reject", "POST",
                           {"expected_revision": veto_row["revision"], "reason": "Fixture owner veto"})["result"]
    assert rejected["status"] == "rejected"
    assert rejected["grade"]["verdict"] == "accept"
    portal.approve(ids["memory"])
    result = absorb(portal.root)
    assert len(result["absorbed"]) == 1
    assert len(latest_records(portal.root / "memory/memory.jsonl")) == 1


def test_dream_schedule_and_idle_context_are_manual_typed_handoffs(portal):
    schedule = portal.get("/schedule")
    fields = {"enabled": input_field("", "checkbox", checked=True), "timezone": input_field("America/New_York"),
              "windows": input_field(json.dumps(schedule["windows"]), json_value=True),
              "min_idle_seconds": input_field(1200, "number"), "max_active_requests": input_field(0, "number")}
    portal.call("/schedule", "PUT", schema="Schedule", fields=fields,
                previous={"expected_revision": schedule["revision"]})
    context = {"at": "2026-10-13T21:00:00-04:00", "idle_seconds": 900, "active_requests": 0}
    waiting = portal.call("/dreams/fixture-dream/check", "POST", context)["result"]
    assert waiting["reasons"] == ["idle_threshold"]
    context["idle_seconds"] = 1200
    ready = portal.call("/dreams/fixture-dream/check", "POST", context)["result"]
    assert ready["eligible"] and ready["next_action"] == "human_confirm"
    assert ready["executes_training"] is False


def test_pocket_copy_absence_empty_selection_and_lexical_package_roundtrip(portal):
    identifier = portal.fixture["candidate_ids"]["pocket"]
    portal.approve(identifier)
    pocket = portal.get("/pockets/aster")
    extracted = portal.call("/pockets/aster/extract", "POST",
                            {"expected_revision": pocket["revision"], "candidate_ids": [identifier]})["result"]
    copied = portal.call("/pockets/aster/copy", "POST", status=201, schema="PocketCopy",
                         fields={"id": input_field("whole-copy"), "fact_ids": input_field("", json_value=True)})
    assert "fact_ids" not in copied["body"] and copied["result"]["facts"] == extracted["facts"]
    portal.call("/pockets/aster/copy", "POST", status=400, schema="PocketCopy",
                fields={"id": input_field("empty-copy"), "fact_ids": input_field("[]", json_value=True)})
    selected = portal.call("/pockets/aster/copy", "POST", status=201, schema="PocketCopy",
                           fields={"id": input_field("chosen-copy"),
                                   "fact_ids": input_field(json.dumps([identifier]), json_value=True)})
    assert len(selected["result"]["facts"]) == 1
    package = portal.call("/pockets/aster/export", representation={"responseText": True})["result"]
    raw = '{"id":"shared-copy","package":' + package + '}'
    imported = portal.call("/pockets/import", "POST", json.loads(raw), status=201,
                           representation={"requestText": raw})["result"]
    assert imported["shared"] and imported["facts"] == extracted["facts"]
    lexical = json.loads(package)
    lexical["facts"][0]["provenance"].update({"version": 1.0, "epsilon": 1e-8, "note": "Café"})
    lexical.pop("sha256")
    lexical["sha256"] = control.digest(lexical)
    raw = json.dumps({"id": "lexical-copy", "package": lexical}, ensure_ascii=False)
    lexical_copy = portal.call("/pockets/import", "POST", json.loads(raw), status=201,
                              representation={"requestText": raw})
    assert lexical_copy["requests"][0]["body"] == raw
    assert lexical_copy["result"]["facts"][0]["provenance"]["version"] == 1.0
    rolled = portal.call("/pockets/aster/rollover", "POST",
                         {"expected_revision": extracted["revision"], "id": "aster-next", "capacity": 2}, status=201)["result"]
    assert rolled["previous"]["sealed"] and rolled["successor"]["facts"] == []


def test_demo_refuses_reuse_and_restores_temporary_authority(tmp_path, monkeypatch):
    root = tmp_path / "fresh"
    monkeypatch.setenv("PREDREAM_MODE", "original-mode")
    with demo_authority(root, "harness") as authority:
        private = authority["token_file"].parent
        receipt = prepare_fixture(root)
        assert receipt["model_calls"] == 0 and receipt["fixture_protocol_responses"] == 8
        before = (root / "queue/candidates.jsonl").read_bytes()
        with pytest.raises(FileExistsError):
            prepare_fixture(root)
        assert before == (root / "queue/candidates.jsonl").read_bytes()
    assert not private.exists()
    assert os.environ["PREDREAM_MODE"] == "original-mode"


def test_remaining_catalog_operations_and_control_with_completed_handoff(portal):
    presets = portal.get("/rules/presets")
    assert len(presets["items"]) == 2
    portal.call("/rules/presets", "POST", {"id": "persona-fixture", "preset": "persona"}, status=201)
    rule = portal.get("/rules/persona-fixture")
    checked = portal.call("/rules/check", "POST", {"user_id": "fixture", "domain": "persona",
                          "source_kind": "manual", "text": "Fictional Aster is a station."})["result"]
    assert checked["selected"]["id"] == rule["id"]
    portal.call("/rules/persona-fixture", "DELETE", {"expected_revision": rule["revision"]})
    profile = portal.get("/policy")
    profile["policy"]["extract"]["probe_temperature"] = None
    updated = portal.call("/policy", "PUT", {"expected_revision": profile["revision"],
                          "policy": profile["policy"]})["result"]
    assert updated["policy"]["extract"]["probe_temperature"] is None
    job = portal.call("/dreams", "POST", {"id": "second-dream", "run_id": "fixture-stage",
                      "not_before": "2026-10-13T19:00:00-04:00"}, status=201)["result"]
    assert portal.get("/dreams/second-dream") == job
    cancelled = portal.call("/dreams/second-dream", "DELETE", {"expected_revision": job["revision"]})["result"]
    assert cancelled["status"] == "cancelled"
    current = portal.get("/control")
    staged_id = portal.fixture["candidate_ids"]["staged"]
    staged = portal.get("/queue/" + staged_id)
    changed = portal.call("/control", "PUT", {"expected_revision": current["revision"], "autonomy": "automatic",
                "auto_routes": ["memory"], "min_score": 0.9})["result"]
    assert changed["autonomy"] == "automatic"
    assert portal.get("/queue/" + staged_id) == staged
    approved = portal.call("/control/approve", "POST", {"expected_revision": changed["revision"],
        "autonomy": "manual", "auto_routes": ["memory"], "min_score": 0.9})["result"]
    assert approved["autonomy"] == "manual"
    portal.call("/pockets", "POST", {"id": "other", "topic": "aster", "capacity": 2}, status=201)
    portal.call("/pockets/merge", "POST", {"id": "merged", "sources": ["aster", "other"], "capacity": 3}, status=201)
    merged = portal.get("/pockets/merged")
    portal.call("/pockets/merged", "DELETE", {"expected_revision": merged["revision"]})


@pytest.mark.parametrize("mode", ["spark", "harness"])
def test_unsealed_autonomy_changes_open_under_the_signed_grant(tmp_path, mode):
    root = tmp_path / "unsealed"
    with demo_authority(root, mode) as authority:
        with running_portal(root, authority, {}) as portal:
            for autonomy in ("supervised", "automatic", "manual"):
                current = portal.get("/control")
                changed = portal.call("/control", "PUT", {"expected_revision": current["revision"],
                    "autonomy": autonomy, "auto_routes": ["memory"], "min_score": 0.95})["result"]
                assert changed["autonomy"] == autonomy and changed["mode"] == mode
                assert changed["revision"] == current["revision"] + 1


@pytest.mark.parametrize("mode", ["spark", "harness"])
def test_default_portal_client_observes_closed_dreams_and_usable_funnel(tmp_path, mode):
    root = tmp_path / "closed"
    with demo_authority(root, mode) as authority:
        fixture = prepare_fixture(root)
        with running_portal(root, authority, fixture) as portal:
            system = portal.get("/system")
            assert system["features"]["dreams"]["status"] == "Under development"
            for path in ("/rules", "/queue", "/pockets", "/control", "/policy"):
                portal.get(path)
            for method, path, operation in (
                ("GET", "/schedule", None), ("PUT", "/schedule", {}),
                ("GET", "/dreams", None), ("POST", "/dreams", {}),
                ("GET", "/dreams/fixture-dream", None),
                ("DELETE", "/dreams/fixture-dream", {}),
                ("POST", "/dreams/fixture-dream/check", {}),
            ):
                result = portal.call(path, method, operation, status=503)
                assert result["error"] == "feature_unavailable: under development"
            assert portal.approve(fixture["candidate_ids"]["pocket"])["status"] == "accepted"
