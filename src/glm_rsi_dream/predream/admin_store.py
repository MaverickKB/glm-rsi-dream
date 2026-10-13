"""Durable local admin state and portable LOOP2 controller operations."""

import copy
import fcntl
import json
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ..files import read_json, write_json
from . import admin_learn, admin_pockets, control
from .absorb import dream_plan
from .admin_schema import OPERATIONS, RULE_FIELDS, SCHEMAS, validate
from .manual import prepared_secret_signal, secret_signal
from .queue import candidate_id, latest_records, record_revision, replace_records
from .timing import now_et


class AdminError(ValueError):
    def __init__(self, status: Any, message: Any) -> None:
        super().__init__(message)
        self.status = status


def initial_state() -> Any:
    return {"schema_version": 1, "revision": 1, "rules": {}, "pockets": {}, "dreams": {},
            "schedule": {"revision": 1, "enabled": False, "timezone": "America/New_York",
                         "windows": [], "min_idle_seconds": 900, "max_active_requests": 0}}


@contextmanager
def admin_lock(root: Any) -> Any:
    directory = Path(root) / "admin"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def fetch(items: Any, identifier: Any) -> Any:
    if identifier not in items:
        raise AdminError(404, "Item was not found")
    return items[identifier]


def available(items: Any, identifier: Any) -> Any:
    if identifier in items:
        raise AdminError(409, "Item id already exists")


def require_revision(item: Any, expected: Any) -> Any:
    if item["revision"] != expected:
        raise AdminError(409, "Revision changed; reload the item")


def queue_view(candidate: Any) -> Any:
    result = copy.deepcopy(candidate)
    result["revision"] = record_revision(candidate)
    return result


def clear_decision(candidate: Any) -> Any:
    result = copy.deepcopy(candidate)
    if result["grade"] is not None:
        result["proposal"] = {"grade": result["grade"], "reason": "Admin edit requires fresh approval",
                              "proposed_et": now_et()}
    result["grade"] = None
    result["status"] = "pending"
    result.pop("approval", None)
    result.pop("admin_veto", None)
    return result


def default_policy(root: Any) -> Any:
    from .extract import EXTRACT_INSTRUCTIONS
    from .grade import validate_policy
    from .novelty import resolve_policy

    extract = resolve_policy()
    extract["extraction_instructions"] = EXTRACT_INSTRUCTIONS
    return {"schema_version": 1, "name": "admin-next-workspace", "extract": extract,
            "grade": validate_policy(), "control": {
                "mode": control.settings_for(root)["mode"], "autonomy": "manual",
                "permission_file": None, "auto_routes": ["memory"], "min_score": 0.9}}


def policy_view(root: Any) -> Any:
    path = Path(root) / "admin/policy-profile.json"
    result = read_json(path) if path.exists() else {"revision": 1, "policy": default_policy(root)}
    active = Path(root) / "policy-config.json"
    result["scope"] = "next_workspace"
    result["active_policy"] = read_json(active)["policy"] if active.exists() else None
    return result


def edit_policy(root: Any, body: Any) -> Any:
    from .grade import validate_policy
    from .novelty import resolve_policy

    current = policy_view(root)
    require_revision(current, body["expected_revision"])
    policy = copy.deepcopy(body["policy"])
    extract = resolve_policy(policy["extract"])
    extract["extraction_instructions"] = policy["extract"].get(
        "extraction_instructions", default_policy(root)["extract"]["extraction_instructions"])
    policy["extract"] = extract
    policy["grade"] = validate_policy(policy["grade"])
    defaults = default_policy(root)["control"]
    defaults.update(policy["control"])
    policy["control"] = defaults
    result: dict[str, Any] = {"revision": current["revision"] + 1, "policy": policy}
    write_json(Path(root) / "admin/policy-profile.json", result)
    return policy_view(root)


def edit_queue(root: Any, identifier: Any, body: Any) -> Any:
    items = {}
    for candidate in latest_records(Path(root) / "queue/candidates.jsonl"):
        items[candidate["id"]] = candidate
    previous = fetch(items, identifier)
    require_revision(queue_view(previous), body["expected_revision"])
    if previous["status"] in ("staged", "absorbed", "learned"):
        raise AdminError(409, "Completed knowledge is immutable; prepare a linked update")
    if set(body) == {"expected_revision"}:
        raise ValueError("Queue edit requires at least one editable field")
    updated = clear_decision(previous)
    review = copy.deepcopy(updated.get("admin_review", {"hold": False, "note": ""}))
    for field in ("hold", "note"):
        if field in body:
            review[field] = body[field]
    updated["admin_review"] = review
    replacements = [updated]
    expected = {identifier: record_revision(previous)}
    replacement_id = None
    if "text" in body or "prompt_evidence" in body:
        if not body.get("text", "").strip() or not body.get("prompt_evidence", "").strip():
            raise ValueError("Knowledge edits require text and literal prompt_evidence")
        prepared = {"text": body["text"], "prompt_evidence": body["prompt_evidence"]}
        if secret_signal(json.dumps(prepared)):
            raise ValueError("Knowledge edit contains credential-shaped input")
        source_ref = "admin-edit:" + identifier + ":" + control.digest(prepared)
        replacement_id = candidate_id(source_ref, prepared["text"])
        replacement = {"id": replacement_id, "created_et": now_et(),
            "source": {"kind": "manual", "ref": source_ref}, **prepared,
            "glm_prior": {"answer": "", "agrees": None}, "novelty": "unmeasured",
            "status": "pending", "grade": None,
            "timings_ms": {"extract": 0, "probe": 0, "grade": 0, "absorb": 0},
            "intake": {"novelty_check": "skipped_manual", "secret_preflight": "clear"},
            "admin_edit": {"previous_id": identifier, "previous_revision": record_revision(previous)},
            "admin_review": {"hold": review["hold"], "note": review["note"]}}
        updated["admin_review"]["hold"] = True
        updated["admin_review"]["replacement_id"] = replacement_id
        if replacement_id not in items:
            replacements.append(replacement)
            expected[replacement_id] = None
    replace_records(Path(root) / "queue/candidates.jsonl", expected, replacements)
    return {"item": queue_view(updated), "replacement_id": replacement_id}


def control_view(root: Any) -> Any:
    result = control.settings_for(root)
    # File locations and signing material are host-owned, not portal values.
    result.pop("permission_file")
    path = Path(root) / "admin/control-settings.json"
    result["revision"] = read_json(path)["revision"] if path.exists() else 1
    return result


def edit_control(root: Any, body: Any, admin: Any, *, renew: Any = False) -> Any:
    current = control_view(root)
    require_revision(current, body["expected_revision"])
    if body["autonomy"] not in admin["autonomy"]:
        raise AdminError(403, "Grant does not authorize the requested autonomy")
    rows = latest_records(Path(root) / "queue/candidates.jsonl")
    expected = {}
    replacements = []
    for row in rows:
        if row["status"] in ("staged", "absorbed", "learned"):
            continue
        if row["status"] in ("accepted", "rejected"):
            expected[row["id"]] = record_revision(row)
            replacements.append(clear_decision(row))
    settings = {"revision": current["revision"] + 1, "autonomy": body["autonomy"],
                "auto_routes": body["auto_routes"], "min_score": body["min_score"]}
    if renew:
        control.renew_permission(root, admin)
    # Clear applied approvals before publishing new settings. Interruption is fail-closed.
    replace_records(Path(root) / "queue/candidates.jsonl", expected, replacements)
    write_json(Path(root) / "admin/control-settings.json", settings)
    return control_view(root)


def build_rule(body: Any) -> Any:
    rule = copy.deepcopy(body)
    for name, schema in RULE_FIELDS.items():
        if name not in rule and "default" in schema:
            rule[name] = copy.deepcopy(schema["default"])
    validate(rule, SCHEMAS["RuleCreate"])
    if (rule["route"] == "pocket") != (rule["pocket"] is not None):
        raise ValueError("Only pocket routes require a pocket name")
    if not rule["name"].strip():
        raise ValueError("Rule name must contain text")
    return dict(rule, revision=1)


def presets() -> Any:
    return [
        {"preset": "companion", "name": "Companion continuity", "domain": "companion",
         "action": "propose", "route": "memory", "notes": "Propose supported companion knowledge for personal memory."},
        {"preset": "persona", "name": "Persona knowledge", "domain": "persona",
         "action": "propose", "route": "pocket", "pocket": "persona",
         "notes": "Propose reusable persona knowledge into its named pocket."},
    ]


def check_rules(rules: Any, context: Any) -> Any:
    matching = []
    for rule in rules.values():
        if not rule["enabled"] or rule["domain"] not in ("general", context["domain"]):
            continue
        if context["user_id"] in rule["exclude_user_ids"]:
            continue
        if rule["user_ids"] and context["user_id"] not in rule["user_ids"]:
            continue
        if context["source_kind"] not in rule["source_kinds"]:
            continue
        if rule["patterns"]:
            found = False
            for pattern in rule["patterns"]:
                if pattern.casefold() in context["text"].casefold():
                    found = True
                    break
            if not found:
                continue
        matching.append(rule)
    matching.sort(key=lambda rule: (rule["priority"], rule["id"]))
    return {"matches": matching, "selected": matching[0] if matching else None,
            "effect": "proposal_policy_only", "collection_enabled": False}


def inside_window(schedule: Any, at: Any) -> Any:
    local = at.astimezone(ZoneInfo(schedule["timezone"]))
    minute = local.hour * 60 + local.minute
    for window in schedule["windows"]:
        start_hour, start_minute = window["start"].split(":")
        end_hour, end_minute = window["end"].split(":")
        start = int(start_hour) * 60 + int(start_minute)
        end = int(end_hour) * 60 + int(end_minute)
        if start < end and local.weekday() in window["days"] and start <= minute < end:
            return True
        if start > end:
            if local.weekday() in window["days"] and minute >= start:
                return True
            previous = (local - timedelta(days=1)).weekday()
            if previous in window["days"] and minute < end:
                return True
    return False


def check_dream(root: Any, schedule: Any, job: Any, context: Any) -> Any:
    at = datetime.fromisoformat(context["at"])
    reasons = []
    plan = None
    if job["status"] == "cancelled":
        reasons.append("cancelled")
    if not schedule["enabled"]:
        reasons.append("schedule_disabled")
    if at < datetime.fromisoformat(job["not_before"]):
        reasons.append("not_due")
    if not inside_window(schedule, at):
        reasons.append("outside_approved_window")
    if context["idle_seconds"] < schedule["min_idle_seconds"]:
        reasons.append("idle_threshold")
    if context["active_requests"] > schedule["max_active_requests"]:
        reasons.append("active_requests")
    try:
        control.permission_for(root, "stage")
        plan = dream_plan(root, job["run_id"])
        if not plan["runs"] or not plan["runs"][0]["groups"]:
            reasons.append("empty_staging")
    except (ValueError, OSError, KeyError) as error:
        reasons.append("handoff_unavailable: " + str(error))
    return {"id": job["id"], "eligible": not reasons, "reasons": reasons, "context": context,
            "context_source": "admin_supplied", "next_action": "human_confirm" if not reasons else "wait",
            "plan": plan, "executes_training": False}


class AdminStore:
    def __init__(self, root: Any, *, client: Any = None, dream_features_enabled: Any = False) -> None:
        if type(dream_features_enabled) is not bool:
            raise ValueError("Dream feature flag must be a boolean")
        self.root = Path(root).resolve()
        self.dream_features_enabled = dream_features_enabled
        self.client = client

    def ensure_available(self, operation: Any) -> Any:
        if operation.startswith(("dreams_", "schedule_")) and not self.dream_features_enabled:
            raise AdminError(503, "feature_unavailable: under development")

    def request(self, operation: Any, body: Any = None, identifier: Any = None) -> Any:
        self.ensure_available(operation)
        body = body or {}
        request_schema = None
        mutating = operation.endswith(("_create", "_edit", "_delete", "_put", "_preset", "_approve",
                                       "_import", "_merge", "_extract", "_copy", "_rollover", "_cancel", "_reject", "_grade"))
        for _, _, name, schema, _ in OPERATIONS:
            if name == operation:
                request_schema = schema
                break
        if request_schema is not None:
            validate(body, SCHEMAS[request_schema])
        if operation == "learn_create" and (prepared_secret_signal(body["content"], body["format"]) or secret_signal(body["source_ref"])):
            raise ValueError("Request contains credential-shaped input")
        if operation in ("learn_create", "queue_edit", "pockets_import", "policy_put") and secret_signal(json.dumps(body)):
            raise ValueError("Request contains credential-shaped input")
        admin: Any = None
        if mutating:
            try:
                _, admin = control.permission_for(self.root, "approve", allow_expired=operation == "control_approve")
            except (ValueError, OSError, KeyError) as error:
                raise AdminError(403, "Admin grant denied: " + str(error)) from error
        with admin_lock(self.root):
            path = self.root / "admin/state.json"
            state = read_json(path) if path.exists() else initial_state()
            result = self.dispatch(state, operation, body, identifier, admin)
            if mutating:
                previous_revision = state["revision"]
                state["revision"] += 1
                state["last_change"] = {"admin_id": admin["admin_id"], "operation": operation, "at": now_et()}
                history = self.root / "admin/history" / f"{state['revision']:08d}.json"
                write_json(history, state)
                write_json(path, state)
                assert state["revision"] == previous_revision + 1
            return result

    def dispatch(self, state: Any, operation: Any, body: Any, identifier: Any, admin: Any) -> Any:
        if operation == "learn_create":
            if body["format"] == "text" and self.client is None:
                raise AdminError(503, "Host GLM client is unavailable")
            policy = policy_view(self.root)["active_policy"] or {}
            result = admin_learn.intake(self.root, body, client=self.client, policy=policy)
            for index, item in enumerate(result["items"]):
                result["items"][index] = queue_view(item)
            return result
        if operation == "system":
            authority = {"available": False}
            try:
                _, grant = control.permission_for(self.root, "approve")
                authority = {"available": True, "admin_id": grant["admin_id"], "expires_et": grant["expires_et"]}
            except (ValueError, OSError, KeyError):
                pass
            return {"control": control_view(self.root), "authority": authority,
                    "collection_enabled": False, "training_execution": False, "pocket_kind": "raw_facts",
                    "features": {"dreams": {"enabled": self.dream_features_enabled,
                        "label": "Manual preparation" if self.dream_features_enabled else "Under development",
                        "status": "Manual preparation" if self.dream_features_enabled else "Under development",
                        "owner": "dream-neurons", "opening_criterion": "Pocket-OFF recall bar passes"}}}
        if operation == "control_get":
            return control_view(self.root)
        if operation == "control_put":
            return edit_control(self.root, body, admin)
        if operation == "control_approve":
            return edit_control(self.root, body, admin, renew=True)
        if operation == "policy_get":
            return policy_view(self.root)
        if operation == "policy_put":
            return edit_policy(self.root, body)
        if operation == "rules_presets":
            return {"items": presets()}
        if operation == "rules_check":
            return check_rules(state["rules"], body)
        if operation == "rules_preset":
            available(state["rules"], body["id"])
            for template in presets():
                if template["preset"] == body["preset"]:
                    prepared = dict(template)
                    prepared.pop("preset")
                    prepared["id"] = body["id"]
                    rule = build_rule(prepared)
                    state["rules"][rule["id"]] = rule
                    return rule
        if operation == "rules_create":
            available(state["rules"], body["id"])
            rule = build_rule(body)
            state["rules"][rule["id"]] = rule
            return rule
        if operation in ("rules_edit", "rules_delete"):
            previous = fetch(state["rules"], identifier)
            require_revision(previous, body["expected_revision"])
            if operation == "rules_delete":
                del state["rules"][identifier]
                return {"removed": identifier}
            fields = dict(previous)
            fields.pop("revision")
            for key, value in body.items():
                if key != "expected_revision":
                    fields[key] = value
            updated = build_rule(fields)
            updated["revision"] = previous["revision"] + 1
            state["rules"][identifier] = updated
            return updated
        if operation.startswith("queue_"):
            candidates = {}
            for candidate in latest_records(self.root / "queue/candidates.jsonl"):
                candidates[candidate["id"]] = candidate
            if operation == "queue_list":
                items = []
                for candidate in candidates.values():
                    items.append(queue_view(candidate))
                return {"items": items}
            candidate = fetch(candidates, identifier)
            if operation == "queue_grade":
                require_revision(queue_view(candidate), body["expected_revision"])
                if candidate["source"]["kind"] != "manual":
                    raise ValueError("Prepared grading requires a manual candidate")
                if candidate["status"] == "pending" and "proposal" not in candidate and self.client is None:
                    raise AdminError(503, "Host GLM client is unavailable")
                policy = policy_view(self.root)["active_policy"] or {}
                result = admin_learn.grade(self.root, candidate, client=self.client, policy=policy)
                result["item"] = queue_view(result["item"])
                return result
            if operation == "queue_get":
                return queue_view(candidate)
            if operation == "queue_edit":
                return edit_queue(self.root, identifier, body)
            if operation in ("queue_approve", "queue_reject"):
                expected = {identifier: body["expected_revision"]}
                if operation == "queue_approve":
                    control.approve(self.root, [identifier], expected_revisions=expected)
                else:
                    control.reject(self.root, [identifier], reason=body["reason"], expected_revisions=expected)
                for row in latest_records(self.root / "queue/candidates.jsonl"):
                    if row["id"] == identifier:
                        return queue_view(row)
        if operation == "schedule_get":
            return state["schedule"]
        if operation == "schedule_put":
            require_revision(state["schedule"], body["expected_revision"])
            ZoneInfo(body["timezone"])
            if body["enabled"] and not body["windows"]:
                raise ValueError("Enabled scheduling requires approved windows")
            for window in body["windows"]:
                if window["start"] == window["end"]:
                    raise ValueError("Approved window must have a distinct start and end")
            schedule = dict(body)
            schedule.pop("expected_revision")
            schedule["revision"] = state["schedule"]["revision"] + 1
            state["schedule"] = schedule
            return schedule
        if operation == "dreams_create":
            available(state["dreams"], body["id"])
            plan = dream_plan(self.root, body["run_id"])
            if not plan["runs"] or not plan["runs"][0]["groups"]:
                raise ValueError("Dream job requires a nonempty sealed staged run")
            job = dict(body, revision=1, status="scheduled", approved_by=admin["admin_id"], approved_et=now_et())
            state["dreams"][job["id"]] = job
            return job
        if operation in ("dreams_check", "dreams_cancel"):
            job = fetch(state["dreams"], identifier)
            if operation == "dreams_check":
                return check_dream(self.root, state["schedule"], job, body)
            require_revision(job, body["expected_revision"])
            updated = dict(job, status="cancelled", revision=job["revision"] + 1)
            state["dreams"][identifier] = updated
            return updated
        if operation == "pockets_create":
            available(state["pockets"], body["id"])
            pocket = admin_pockets.new_pocket(body["id"], body["topic"], body["capacity"])
            state["pockets"][pocket["id"]] = pocket
            return pocket
        if operation == "pockets_import":
            available(state["pockets"], body["id"])
            pocket = admin_pockets.import_pocket(body["id"], body["package"])
            state["pockets"][pocket["id"]] = pocket
            return pocket
        if operation == "pockets_merge":
            available(state["pockets"], body["id"])
            sources = []
            for source_id in body["sources"]:
                sources.append(fetch(state["pockets"], source_id))
            pocket = admin_pockets.merge_pockets(body["id"], sources, body["capacity"])
            state["pockets"][pocket["id"]] = pocket
            return pocket
        if operation in ("pockets_export", "pockets_copy", "pockets_rollover", "pockets_extract", "pockets_delete"):
            pocket = fetch(state["pockets"], identifier)
            if operation == "pockets_export":
                return admin_pockets.export_pocket(pocket)
            if operation == "pockets_delete":
                require_revision(pocket, body["expected_revision"])
                del state["pockets"][identifier]
                return {"removed": identifier}
            if operation == "pockets_extract":
                candidates = {}
                for candidate in latest_records(self.root / "queue/candidates.jsonl"):
                    candidates[candidate["id"]] = candidate
                selected = []
                for candidate_id in body["candidate_ids"]:
                    selected.append(fetch(candidates, candidate_id))
                updated = admin_pockets.extract_facts(self.root, pocket, selected, body["expected_revision"])
                state["pockets"][identifier] = updated
                return updated
            available(state["pockets"], body["id"])
            if operation == "pockets_copy":
                copied = admin_pockets.copy_pocket(pocket, body["id"], body.get("fact_ids"))
                state["pockets"][copied["id"]] = copied
                return copied
            previous, successor = admin_pockets.rollover_pocket(pocket, body["id"], body["capacity"], body["expected_revision"])
            state["pockets"][identifier] = previous
            state["pockets"][successor["id"]] = successor
            return {"previous": previous, "successor": successor}
        for collection in ("rules", "pockets", "dreams"):
            if operation == collection + "_list":
                return {"items": list(state[collection].values())}
            if operation == collection + "_get":
                return fetch(state[collection], identifier)
        raise AdminError(404, "Operation was not found")
