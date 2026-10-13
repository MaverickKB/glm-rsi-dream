"""Real local HTTP workflows, authority boundaries and portable admin state."""

import copy
import json
import threading
from importlib.resources import files
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from glm_rsi_dream.predream import control
from glm_rsi_dream.predream.absorb import stage
from glm_rsi_dream.predream.admin_api import create_server, read_token
from glm_rsi_dream.predream.admin_schema import openapi_spec
from glm_rsi_dream.predream.admin_store import AdminStore, inside_window
from glm_rsi_dream.predream.grade import grade_pending
from glm_rsi_dream.predream.policy import load_policy, record_policy
from glm_rsi_dream.predream.queue import (
    append_record,
    candidate_id,
    latest_records,
    record_revision,
    replace_records,
)


class API:
    def __init__(self, root, token_file):
        self.root = root
        self.token_file = token_file
        self.server = create_server(root, token_file, port=0, allowed_origins=["http://localhost:5173"],
                                    dream_features_enabled=True)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01})
        self.thread.start()
        self.url = "http://127.0.0.1:" + str(self.server.server_port)

    def close(self):
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()

    def call(self, method, path, body=None, *, auth=True, headers=None):
        request_headers = {"Content-Type": "application/json"}
        if auth:
            request_headers["Authorization"] = "Bearer " + self.token_file.read_text().strip()
        if headers:
            request_headers.update(headers)
        data = json.dumps(body).encode() if body is not None else None
        request = Request(self.url + path, data=data, headers=request_headers, method=method)
        try:
            response = urlopen(request, timeout=5)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.load(response), dict(response.headers)

    def ok(self, method, path, body=None, status=200):
        actual, result, _ = self.call(method, "/api/v1" + path, body)
        assert actual == status, result
        return result


@pytest.fixture(params=["spark", "harness"])
def api(tmp_path, monkeypatch, request):
    root = tmp_path / "run"
    key = tmp_path / "signing.key"
    token = tmp_path / "http.token"
    permission = tmp_path / "permission.json"
    control.initialize_key(key)
    control.initialize_key(token)
    monkeypatch.setenv("PREDREAM_ADMIN_KEY_FILE", str(key))
    monkeypatch.setenv("PREDREAM_ADMIN_PERMISSION_FILE", str(permission))
    monkeypatch.setenv("PREDREAM_MODE", request.param)
    monkeypatch.setenv("PREDREAM_AUTONOMY", "manual")
    control.issue_permission(permission, admin_id="fixture:api-admin", root_scope=root,
                             modes=[request.param], autonomy=["manual", "supervised", "automatic"])
    instance = API(root, token)
    instance.permission = permission
    instance.mode = request.param
    yield instance
    instance.close()


def candidate(root, text="Fictional Iona's hatch is cobalt.", route="pocket"):
    row = {"id": candidate_id("api-fixture:" + text, text), "created_et": "2026-10-13T01:00:00-04:00",
           "source": {"kind": "prompt", "ref": "api-fixture:" + text}, "text": text,
           "prompt_evidence": text, "glm_prior": {"answer": "Unknown", "agrees": None},
           "novelty": "new", "status": "pending", "grade": None,
           "timings_ms": {"extract": 0, "probe": 0, "grade": 0, "absorb": 0}}
    grade = {"verdict": "accept", "route": route, "pocket": "iona" if route == "pocket" else None,
             "reason": "Synthetic model proposal for an explicitly supplied fixture.", "score": 0.95, "grader": "glm"}
    queued = control.apply_proposal(root, row, grade)
    append_record(root / "queue/candidates.jsonl", queued)
    return row["id"]


def approve(api, identifier):
    row = api.ok("GET", "/queue/" + identifier)
    return api.ok("POST", "/queue/" + identifier + "/approve", {"expected_revision": row["revision"]})


def rule(identifier="explicit", **changes):
    result = {"id": identifier, "name": "Explicit fixture knowledge", "domain": "general",
              "action": "propose", "route": "memory"}
    result.update(changes)
    return result


def test_bearer_and_signed_authority_are_independent(api):
    status, _, _ = api.call("GET", "/api/v1/queue", auth=False)
    assert status == 401
    status, _, _ = api.call("GET", "/openapi.json", auth=False)
    assert status == 200
    assert api.ok("GET", "/system")["control"]["mode"] == api.mode
    grant = json.loads(api.permission.read_text())
    grant["permission"]["admin_id"] = "forged"
    api.permission.write_text(json.dumps(grant))
    status, _, _ = api.call("POST", "/api/v1/rules", rule())
    assert status == 403
    assert not (api.root / "admin/state.json").exists()
    assert api.ok("GET", "/rules")["items"] == []
    assert api.ok("GET", "/system")["authority"]["available"] is False


def test_rules_crud_presets_matching_and_history(api):
    created = api.ok("POST", "/rules", rule(exclude_user_ids=["excluded"], patterns=["hatch"], priority=10), 201)
    assert created["enabled"] and created["revision"] == 1
    for name in ("companion", "persona"):
        preset = api.ok("POST", "/rules/presets", {"id": name, "preset": name}, 201)
        assert preset["domain"] == name
    context = {"user_id": "owner", "domain": "companion", "source_kind": "manual", "text": "Iona's HATCH is cobalt."}
    assert api.ok("POST", "/rules/check", context)["selected"]["id"] == "explicit"
    context["user_id"] = "excluded"
    assert api.ok("POST", "/rules/check", context)["selected"]["id"] == "companion"
    changed = api.ok("PATCH", "/rules/explicit", {"expected_revision": 1, "enabled": False, "notes": "Reviewed"})
    assert changed["revision"] == 2
    before = (api.root / "admin/state.json").read_bytes()
    status, _, _ = api.call("PATCH", "/api/v1/rules/explicit", {"expected_revision": 1, "enabled": True})
    assert status == 409 and (api.root / "admin/state.json").read_bytes() == before
    api.ok("DELETE", "/rules/explicit", {"expected_revision": 2})
    assert len(api.ok("GET", "/rules")["items"]) == 2
    histories = list((api.root / "admin/history").glob("*.json"))
    assert any("explicit" in json.loads(path.read_text())["rules"] for path in histories)


def test_rule_can_ignore_credential_assignment_patterns(api):
    api.ok("POST", "/rules", rule("drop-password", action="ignore", patterns=["password="]), 201)
    result = api.ok("POST", "/rules/check", {"user_id": "fixture", "domain": "general",
                   "source_kind": "manual", "text": "password=fictional-test-only"})
    assert result["selected"]["action"] == "ignore"


def test_admin_body_limit_has_an_observable_opening_path(api):
    api.server.max_body_bytes = 64
    status, _, _ = api.call("POST", "/api/v1/rules", rule())
    assert status == 413 and not (api.root / "admin/state.json").exists()
    api.server.max_body_bytes = 4096
    assert api.ok("POST", "/rules", rule(), 201)["id"] == "explicit"


def test_queue_hold_is_enforced_by_controller_and_staging(api):
    identifier = candidate(api.root)
    accepted = approve(api, identifier)
    assert accepted["status"] == "accepted"
    result = api.ok("PATCH", "/queue/" + identifier,
                    {"expected_revision": accepted["revision"], "hold": True, "note": "Needs owner review"})
    held = result["item"]
    assert held["status"] == "pending" and held["grade"] is None
    status, _, _ = api.call("POST", "/api/v1/queue/" + identifier + "/approve", {"expected_revision": held["revision"]})
    assert status == 400
    assert stage(api.root, "held")["items"] == []
    row = latest_records(api.root / "queue/candidates.jsonl")[0]
    proposed = control.apply_proposal(api.root, row, row["proposal"]["grade"])
    assert proposed["status"] == "pending"
    restored = api.ok("PATCH", "/queue/" + identifier, {"expected_revision": held["revision"], "hold": False})["item"]
    status, _, _ = api.call("PATCH", "/api/v1/queue/" + identifier, {"expected_revision": held["revision"], "note": "stale"})
    assert status == 409
    assert restored["admin_review"]["hold"] is False
    approve(api, identifier)
    assert len(stage(api.root, "released")["items"]) == 1
    sealed = api.ok("GET", "/queue/" + identifier)
    status, _, _ = api.call("PATCH", "/api/v1/queue/" + identifier, {"expected_revision": sealed["revision"], "hold": True})
    assert status == 409


def test_knowledge_edit_reenters_the_manual_glm_path(api):
    identifier = candidate(api.root)
    old = approve(api, identifier)
    text = "Fictional Iona's hatch is copper."
    result = api.ok("PATCH", "/queue/" + identifier,
                    {"expected_revision": old["revision"], "text": text, "prompt_evidence": text})
    new = api.ok("GET", "/queue/" + result["replacement_id"])
    assert new["source"]["kind"] == "manual" and new["novelty"] == "unmeasured"
    assert "review" not in new and "proposal" not in new and "approval" not in new
    assert result["item"]["admin_review"]["hold"] is True
    answers = iter([
        {"safe": True, "kind": "new", "targets": [], "reason": "Explicit synthetic fixture review"},
        {"verdict": "accept", "route": "pocket", "pocket": "iona", "reason": "Prepared supported fixture", "score": 0.95},
    ])

    def synthetic_model(_):
        return json.dumps(next(answers))

    graded = grade_pending(api.root, synthetic_model, candidate_id=new["id"])
    assert not graded["errors"]
    proposed = api.ok("GET", "/queue/" + new["id"])
    assert proposed["status"] == "pending" and proposed["review"]["safe"]
    approve(api, new["id"])
    manifest = stage(api.root, "edited")
    assert [item["id"] for item in manifest["items"]] == [new["id"]]


def test_autonomy_changes_are_permission_bounded_and_invalidate_approvals(api):
    identifier = candidate(api.root)
    accepted = approve(api, identifier)
    settings = api.ok("PUT", "/control", {"expected_revision": 1, "autonomy": "automatic", "auto_routes": ["memory"], "min_score": 0.9})
    assert settings["mode"] == api.mode and settings["autonomy"] == "automatic"
    pending = api.ok("GET", "/queue/" + identifier)
    assert pending["status"] == "pending" and "approval" not in pending
    approved_snapshot = dict(accepted)
    approved_snapshot.pop("revision")
    with pytest.raises(ValueError, match="Policy changed"):
        control.authorize_handoff(api.root, approved_snapshot)
    second = candidate(api.root, "Fictional Iona has two engines.")
    assert api.ok("GET", "/queue/" + second)["status"] == "accepted"
    stage(api.root, "automatic")
    staged = api.ok("GET", "/queue/" + second)
    manifest = (api.root / "staged/automatic/manifest.json").read_bytes()
    changed = api.ok("PUT", "/control", {"expected_revision": 2, "autonomy": "manual", "auto_routes": [], "min_score": 1})
    assert changed["autonomy"] == "manual"
    assert api.ok("GET", "/queue/" + second) == staged
    assert (api.root / "staged/automatic/manifest.json").read_bytes() == manifest
    future = candidate(api.root, "Fictional Iona's next hatch is silver.")
    assert api.ok("GET", "/queue/" + future)["status"] == "pending"


def test_admin_session_approval_renews_only_expiry_and_applies_level(api):
    grant = json.loads(api.permission.read_text())
    grant["permission"]["expires_et"] = "2020-01-01T00:00:00-05:00"
    grant["signature"] = control.signature(grant["permission"], control.admin_key())
    api.permission.write_text(json.dumps(grant))
    previous = api.permission.read_bytes()
    body = {"expected_revision": 1, "autonomy": "automatic", "auto_routes": ["memory"], "min_score": 0.9}
    assert api.call("PUT", "/api/v1/control", body)[0] == 403
    assert api.call("POST", "/api/v1/control/approve", body, auth=False)[0] == 401
    changed = api.ok("POST", "/control/approve", body)
    assert changed["autonomy"] == "automatic" and changed["revision"] == 2
    renewed = json.loads(api.permission.read_text())
    assert renewed["signature"] == control.signature(renewed["permission"], control.admin_key())
    for name, value in grant["permission"].items():
        if name != "expires_et":
            assert renewed["permission"][name] == value
    assert control.permission_for(api.root, "approve")[1] == renewed["permission"]
    history = list((api.root / "admin/permission-history").glob("*.json"))
    assert len(history) == 1 and history[0].read_bytes() == previous
    assert api.permission.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("problem", ["signature", "scope", "mode", "action", "level", "revision"])
def test_admin_session_approval_cannot_expand_or_forge_access(api, problem):
    grant = json.loads(api.permission.read_text())
    grant["permission"]["expires_et"] = "2020-01-01T00:00:00-05:00"
    if problem == "scope":
        grant["permission"]["root_scope"] = str(api.root / "other")
    elif problem == "mode":
        grant["permission"]["modes"] = ["harness" if api.mode == "spark" else "spark"]
    elif problem == "action":
        grant["permission"]["actions"] = ["stage"]
    elif problem == "level":
        grant["permission"]["autonomy"] = ["manual"]
    grant["signature"] = control.signature(grant["permission"], control.admin_key())
    if problem == "signature":
        grant["signature"] = "0" * 64
    api.permission.write_text(json.dumps(grant))
    previous = api.permission.read_bytes()
    body = {"expected_revision": 999 if problem == "revision" else 1,
            "autonomy": "automatic", "auto_routes": ["memory"], "min_score": 0.9}
    status, _, _ = api.call("POST", "/api/v1/control/approve", body)
    assert status == (409 if problem == "revision" else 403)
    assert api.permission.read_bytes() == previous
    assert not (api.root / "admin/control-settings.json").exists()
    assert not (api.root / "admin/permission-history").exists()


def test_restoring_control_values_cannot_revive_an_old_approval(api):
    original_settings = {"expected_revision": 1, "autonomy": "automatic", "auto_routes": ["memory"], "min_score": 0.9}
    first_settings = api.ok("PUT", "/control", original_settings)
    identifier = candidate(api.root)
    approved_snapshot = api.ok("GET", "/queue/" + identifier)
    approved_snapshot.pop("revision")
    control.authorize_handoff(api.root, approved_snapshot)
    temporary = api.ok("PUT", "/control", dict(original_settings, expected_revision=first_settings["revision"], autonomy="manual"))
    restored = api.ok("PUT", "/control", dict(original_settings, expected_revision=temporary["revision"]))
    assert restored["autonomy"] == "automatic"
    with pytest.raises(ValueError, match="Policy changed"):
        control.authorize_handoff(api.root, approved_snapshot)
    assert api.ok("GET", "/queue/" + identifier)["status"] == "pending"
    assert approve(api, identifier)["status"] == "accepted"


def test_schedule_overnight_boundaries_idle_and_sealed_plan(api):
    identifier = candidate(api.root)
    approve(api, identifier)
    stage(api.root, "prepared")
    schedule = api.ok("PUT", "/schedule", {"expected_revision": 1, "enabled": True,
        "timezone": "America/New_York", "windows": [{"days": [0], "start": "22:00", "end": "02:00"}],
        "min_idle_seconds": 900, "max_active_requests": 0})
    assert schedule["revision"] == 2
    job = api.ok("POST", "/dreams", {"id": "night", "run_id": "prepared", "not_before": "2026-10-12T21:00:00-04:00"}, 201)
    check = {"at": "2026-10-13T01:00:00-04:00", "idle_seconds": 900, "active_requests": 0}
    ready = api.ok("POST", "/dreams/night/check", check)
    assert ready["eligible"] and ready["next_action"] == "human_confirm"
    assert ready["plan"]["human_confirm_required"] and not ready["executes_training"]
    for key, value, reason in (("idle_seconds", 899, "idle_threshold"), ("active_requests", 1, "active_requests"),
                               ("at", "2026-10-13T02:00:00-04:00", "outside_approved_window")):
        failed = api.ok("POST", "/dreams/night/check", dict(check, **{key: value}))
        assert not failed["eligible"] and reason in failed["reasons"]
    cancelled = api.ok("DELETE", "/dreams/night", {"expected_revision": job["revision"]})
    assert cancelled["status"] == "cancelled"
    assert "cancelled" in api.ok("POST", "/dreams/night/check", check)["reasons"]
    assert not list(api.root.glob("training*"))


def test_pocket_extract_export_import_copy_merge_rollover_and_removal(api):
    first = candidate(api.root)
    second = candidate(api.root, "Fictional Iona has two engines.")
    approve(api, first)
    approve(api, second)
    for name, identifier in (("first", first), ("second", second)):
        api.ok("POST", "/pockets", {"id": name, "topic": "iona", "capacity": 1}, 201)
        full = api.ok("POST", "/pockets/" + name + "/extract", {"expected_revision": 1, "candidate_ids": [identifier]})
        assert len(full["facts"]) == 1
    package = api.ok("GET", "/pockets/first/export")
    shared = api.ok("POST", "/pockets/import", {"id": "shared", "package": package}, 201)
    assert shared["shared"] and shared["facts"] == package["facts"]
    copied = api.ok("POST", "/pockets/first/copy", {"id": "copied", "fact_ids": [first]}, 201)
    assert copied["facts"] == package["facts"]
    merged = api.ok("POST", "/pockets/merge", {"id": "merged", "sources": ["first", "second", "copied"], "capacity": 2}, 201)
    assert len(merged["facts"]) == 2 and merged["origin"]["operation"] == "merge"
    rolled = api.ok("POST", "/pockets/first/rollover", {"expected_revision": 2, "id": "next", "capacity": 3}, 201)
    assert rolled["previous"]["sealed"] and rolled["successor"]["facts"] == []
    assert rolled["previous"]["facts"] == package["facts"]
    status, _, _ = api.call("POST", "/api/v1/pockets/first/extract", {"expected_revision": 3, "candidate_ids": [second]})
    assert status == 400
    api.ok("DELETE", "/pockets/shared", {"expected_revision": 1})
    assert len(api.ok("GET", "/pockets")["items"]) == 5
    assert AdminStore(api.root).request("pockets_get", identifier="merged")["facts"] == merged["facts"]


def test_pocket_validation_has_no_partial_changes(api):
    identifier = candidate(api.root)
    api.ok("POST", "/pockets", {"id": "unapproved", "topic": "iona", "capacity": 1}, 201)
    status, _, _ = api.call("POST", "/api/v1/pockets/unapproved/extract", {"expected_revision": 1, "candidate_ids": [identifier]})
    assert status == 400 and api.ok("GET", "/pockets/unapproved")["facts"] == []
    approve(api, identifier)
    api.ok("POST", "/pockets/unapproved/extract", {"expected_revision": 1, "candidate_ids": [identifier]})
    package = api.ok("GET", "/pockets/unapproved/export")
    broken = copy.deepcopy(package)
    broken["facts"][0]["statement"] = "Changed without a matching checksum"
    status, _, _ = api.call("POST", "/api/v1/pockets/import", {"id": "broken", "package": broken})
    assert status == 400
    api.ok("POST", "/pockets", {"id": "other", "topic": "other", "capacity": 1}, 201)
    for sources, capacity in ((["unapproved", "other"], 2), (["unapproved", "other"], 1)):
        status, _, _ = api.call("POST", "/api/v1/pockets/merge", {"id": "bad", "sources": sources, "capacity": capacity})
        assert status == 400
    assert len(api.ok("GET", "/pockets")["items"]) == 2


def test_closed_shapes_cors_and_sensitive_edit_preflight(api):
    status, _, _ = api.call("POST", "/api/v1/rules", dict(rule(), command="sh"))
    assert status == 400
    status, _, _ = api.call("POST", "/api/v1/rules", rule(), headers={"Origin": "http://untrusted.example"})
    assert status == 403
    status, _, headers = api.call("OPTIONS", "/api/v1/rules", auth=False, headers={"Origin": "http://localhost:5173"})
    assert status == 200 and headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
    status, _, _ = api.call("GET", "/api/v1/queue?token=anything")
    assert status == 400
    identifier = candidate(api.root)
    row = api.ok("GET", "/queue/" + identifier)
    before = (api.root / "queue/candidates.jsonl").read_bytes()
    secret = "api_key=sk-" + "x" * 24
    status, _, _ = api.call("PATCH", "/api/v1/queue/" + identifier,
                          {"expected_revision": row["revision"], "text": secret, "prompt_evidence": secret})
    assert status == 400 and (api.root / "queue/candidates.jsonl").read_bytes() == before


def test_admin_veto_preserves_glm_grade_and_has_a_reopening_path(api):
    identifier = candidate(api.root)
    proposed = api.ok("GET", "/queue/" + identifier)
    raw_grade = proposed["proposal"]["grade"]
    rejected = api.ok("POST", "/queue/" + identifier + "/reject",
                     {"expected_revision": proposed["revision"], "reason": "Owner declines this learning"})
    assert rejected["status"] == "rejected" and rejected["grade"] == raw_grade
    assert rejected["admin_veto"]["reason"] == "Owner declines this learning"
    assert rejected["approval"]["decision"]["admin_verdict"] == "reject"
    forged_status = dict(rejected, status="accepted")
    forged_status.pop("revision")
    with pytest.raises(ValueError, match="Admin rejected"):
        control.authorize_handoff(api.root, forged_status)
    reopened = api.ok("PATCH", "/queue/" + identifier,
                     {"expected_revision": rejected["revision"], "hold": False, "note": "Owner reopened review"})["item"]
    assert reopened["status"] == "pending" and "admin_veto" not in reopened
    assert approve(api, identifier)["status"] == "accepted"


def test_next_policy_exposes_all_knobs_and_preserves_active_experiment(api):
    draft = api.ok("GET", "/policy")
    assert len(draft["policy"]["extract"]) == 8
    assert len(draft["policy"]["grade"]) == 4
    assert len(draft["policy"]["control"]) == 5
    active = copy.deepcopy(draft["policy"])
    active["name"] = "immutable-active-fixture"
    active["control"]["permission_file"] = str(api.permission)
    record_policy(api.root, active)
    before = (api.root / "policy-config.json").read_bytes()
    policy = copy.deepcopy(draft["policy"])
    policy["name"] = "next-experiment"
    policy["extract"].update(probe_samples=3, probe_temperature=0.7, agreement_threshold=0.66)
    policy["grade"]["routing_rules"] = "Route supported owner knowledge to a named pocket."
    saved = api.ok("PUT", "/policy", {"expected_revision": 1, "policy": policy})
    assert saved["revision"] == 2 and saved["scope"] == "next_workspace"
    assert saved["active_policy"] == active
    assert (api.root / "policy-config.json").read_bytes() == before
    exported = api.root / "exported-policy.json"
    exported.write_text(json.dumps(saved["policy"]))
    assert load_policy(exported)["extract"]["probe_samples"] == 3
    assert AdminStore(api.root).request("policy_get")["policy"] == saved["policy"]
    bad = copy.deepcopy(saved["policy"])
    bad["extract"]["agreement_threshold"] = 0
    status, _, _ = api.call("PUT", "/api/v1/policy", {"expected_revision": 2, "policy": bad})
    assert status == 400 and api.ok("GET", "/policy")["revision"] == 2


@pytest.mark.parametrize("problem", ["expired", "mode", "scope", "action"])
def test_authentic_but_inapplicable_grants_deny_mutation(api, problem):
    grant = json.loads(api.permission.read_text())
    payload = grant["permission"]
    if problem == "expired":
        payload["expires_et"] = "2020-01-01T00:00:00Z"
    elif problem == "mode":
        payload["modes"] = ["harness" if api.mode == "spark" else "spark"]
    elif problem == "scope":
        payload["root_scope"] = str(api.root.parent / "another-run")
    else:
        payload["actions"] = ["stage"]
    grant["signature"] = control.signature(payload, control.admin_key())
    api.permission.write_text(json.dumps(grant))
    status, _, _ = api.call("POST", "/api/v1/rules", rule())
    assert status == 403 and not (api.root / "admin/state.json").exists()


def test_capacity_and_duplicate_fact_conflicts_preserve_all_pockets(api):
    ids = []
    for text in ("Fictional Iona's hatch is cobalt.", "Fictional Iona has two engines."):
        identifier = candidate(api.root, text)
        approve(api, identifier)
        ids.append(identifier)
    api.ok("POST", "/pockets", {"id": "small", "topic": "iona", "capacity": 1}, 201)
    status, _, _ = api.call("POST", "/api/v1/pockets/small/extract", {"expected_revision": 1, "candidate_ids": ids})
    assert status == 400 and api.ok("GET", "/pockets/small")["facts"] == []
    api.ok("POST", "/pockets/small/extract", {"expected_revision": 1, "candidate_ids": [ids[0]]})
    package = api.ok("GET", "/pockets/small/export")
    package["facts"][0]["statement"] = "A contradictory synthetic catalog value."
    unsigned = dict(package)
    unsigned.pop("sha256")
    package["sha256"] = control.digest(unsigned)
    api.ok("POST", "/pockets/import", {"id": "conflicting", "package": package}, 201)
    before = (api.root / "admin/state.json").read_bytes()
    status, _, _ = api.call("POST", "/api/v1/pockets/merge", {"id": "bad", "sources": ["small", "conflicting"], "capacity": 2})
    assert status == 400 and (api.root / "admin/state.json").read_bytes() == before


def test_empty_missing_or_corrupted_dream_handoffs_never_become_ready(api):
    stage(api.root, "empty")
    status, _, _ = api.call("POST", "/api/v1/dreams", {"id": "empty", "run_id": "empty", "not_before": "2026-10-13T00:00:00Z"})
    assert status == 400
    identifier = candidate(api.root)
    approve(api, identifier)
    stage(api.root, "nonempty")
    api.ok("POST", "/dreams", {"id": "nonempty", "run_id": "nonempty", "not_before": "2026-10-13T00:00:00Z"}, 201)
    manifest = api.root / "staged/nonempty/manifest.json"
    value = json.loads(manifest.read_text())
    value["items"][0]["text"] = "Tampered metadata"
    manifest.write_text(json.dumps(value))
    result = api.ok("POST", "/dreams/nonempty/check", {"at": "2026-10-13T01:00:00-04:00", "idle_seconds": 900, "active_requests": 0})
    assert not result["eligible"]
    assert any(reason.startswith("handoff_unavailable") for reason in result["reasons"])


def test_private_token_and_no_nonlocal_listener(tmp_path):
    token = tmp_path / "token"
    control.initialize_key(token)
    assert read_token(token)
    with pytest.raises(ValueError, match="loopback"):
        create_server(tmp_path, token, host="0.0.0.0", port=0)
    token.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        create_server(tmp_path, token, port=0)


def test_queue_compare_batch_rejects_stale_state_without_partial_write(api):
    first = candidate(api.root)
    second = candidate(api.root, "Fictional Iona has two engines.")
    rows = latest_records(api.root / "queue/candidates.jsonl")
    expected = {}
    for row in rows:
        expected[row["id"]] = record_revision(row)
    original = api.ok("GET", "/queue/" + second)
    api.ok("PATCH", "/queue/" + second, {"expected_revision": original["revision"], "hold": True})
    before = (api.root / "queue/candidates.jsonl").read_bytes()
    with pytest.raises(ValueError, match="revision changed"):
        replace_records(api.root / "queue/candidates.jsonl", expected, rows)
    assert (api.root / "queue/candidates.jsonl").read_bytes() == before
    first_revision = api.ok("GET", "/queue/" + first)["revision"]
    status, _, _ = api.call("POST", "/api/v1/queue/" + first + "/approve", {"expected_revision": "0" * 64})
    assert status == 409 and api.ok("GET", "/queue/" + first)["revision"] == first_revision


def test_overnight_window_uses_start_weekday_through_dst():
    from datetime import datetime

    schedule = {"timezone": "America/New_York", "windows": [{"days": [5], "start": "22:00", "end": "02:00"}]}
    for value in ("2026-11-01T01:30:00-04:00", "2026-11-01T01:30:00-05:00"):
        assert inside_window(schedule, datetime.fromisoformat(value))
    assert not inside_window(schedule, datetime.fromisoformat("2026-11-01T02:00:00-05:00"))


def test_packaged_openapi_matches_the_running_contract():
    path = files("glm_rsi_dream").joinpath("resources/rsi-admin-openapi.json")
    assert json.loads(path.read_text()) == openapi_spec()
    assert openapi_spec()["security"] == [{"AdminBearer": []}]
