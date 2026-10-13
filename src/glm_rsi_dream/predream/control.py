"""Host-admin authority for portable funnel proposals, decisions and staging."""

import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..files import write_json
from .queue import latest_records, record_revision, replace_records
from .timing import now_et


def digest(value: Any) -> Any:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def signature(value: Any, key: Any) -> Any:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hmac.new(key, raw, hashlib.sha256).hexdigest()


def admin_key() -> Any:
    filename = os.environ.get("PREDREAM_ADMIN_KEY_FILE")
    if not filename:
        raise ValueError("Admin host must configure PREDREAM_ADMIN_KEY_FILE")
    path = Path(filename)
    metadata = path.stat()
    if metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
        raise ValueError("Admin key must belong to the host user with mode0600")
    key = bytes.fromhex(path.read_text().strip())
    if len(key) < 32:
        raise ValueError("Admin signing key is too short")
    return key


def initialize_key(path: Any) -> Any:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as file:
        file.write(secrets.token_hex(32) + "\n")
    return {"key_file": str(path), "owner_uid": os.getuid()}


def issue_permission(path: Any, *, admin_id: Any, root_scope: Any, modes: Any, autonomy: Any, hours: Any = 16, actions: Any = None) -> Any:
    if not admin_id or hours <= 0:
        raise ValueError("Admin identity and positive permission lifetime required")
    if not modes or not set(modes) <= {"spark", "harness"}:
        raise ValueError("Permission modes must be spark or harness")
    if not autonomy or not set(autonomy) <= {"manual", "supervised", "automatic"}:
        raise ValueError("Permission autonomy must be manual, supervised or automatic")
    default_actions = ["decide", "route", "stage", "approve", "catalog"]
    allowed_actions = default_actions + ["dream_register", "dream_start", "dream_qualify",
                                         "dream_activate", "dream_retire", "dream_rollback"]
    if actions is None:
        actions = default_actions
    if not actions or not set(actions) <= set(allowed_actions):
        raise ValueError("Unknown or empty admin actions")
    expires = datetime.fromisoformat(now_et()) + timedelta(hours=hours)
    payload = {"schema_version": 1, "admin_id": admin_id, "root_scope": str(Path(root_scope).resolve()),
               "modes": modes, "autonomy": autonomy, "actions": actions,
               "expires_et": expires.isoformat()}
    grant = {"permission": payload, "signature": signature(payload, admin_key())}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as file:
        file.write(json.dumps(grant, indent=2) + "\n")
    return payload


def settings_for(root: Any) -> Any:
    path = Path(root) / "policy-config.json"
    settings: dict[str, Any] = {}
    if path.exists():
        settings = json.loads(path.read_text())["policy"].get("control", {})
    result: dict[str, Any] = {"mode": os.environ.get("PREDREAM_MODE", "harness"),
              "autonomy": os.environ.get("PREDREAM_AUTONOMY", "manual"),
              "permission_file": os.environ.get("PREDREAM_ADMIN_PERMISSION_FILE"),
              "auto_routes": ["memory"], "min_score": 0.9}
    for name, value in settings.items():
        if name not in result:
            raise ValueError("Unknown control setting: " + name)
        result[name] = value
    override = Path(root) / "admin/control-settings.json"
    if override.exists():
        values = json.loads(override.read_text())
        for name in ("autonomy", "auto_routes", "min_score"):
            result[name] = values[name]
    if result["mode"] not in ("spark", "harness") or result["autonomy"] not in ("manual", "supervised", "automatic"):
        raise ValueError("Invalid deployment mode or autonomy level")
    host_mode = os.environ.get("PREDREAM_MODE")
    if host_mode is not None and result["mode"] != host_mode:
        raise ValueError("Configured mode differs from the admin-controlled host mode")
    if not isinstance(result["auto_routes"], list) or not set(result["auto_routes"]) <= {"weights", "pocket", "memory"}:
        raise ValueError("Supervised auto_routes must use queue destinations")
    if type(result["min_score"]) not in (int, float) or not 0 <= result["min_score"] <= 1:
        raise ValueError("Supervised min_score must be between zero and one")
    return result


def policy_identity(root: Any, settings: Any) -> Any:
    path = Path(root) / "policy-config.json"
    identity = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else digest(settings)
    override = Path(root) / "admin/control-settings.json"
    if override.exists():
        revision = json.loads(override.read_text())["revision"]
        identity = digest({"base": identity, "control": settings, "control_revision": revision})
    return identity


def permission_for(root: Any, action: Any, *, allow_expired: Any = False) -> Any:
    settings = settings_for(root)
    if not settings["permission_file"]:
        raise ValueError("Admin permission is unavailable; proposals require explicit approval")
    grant = json.loads(Path(settings["permission_file"]).read_text())
    payload = grant["permission"]
    if not hmac.compare_digest(grant["signature"], signature(payload, admin_key())):
        raise ValueError("Admin permission signature is invalid")
    expires = datetime.fromisoformat(payload["expires_et"])
    if expires.tzinfo is None:
        raise ValueError("Admin permission has no expiry timezone")
    if not allow_expired and expires <= datetime.fromisoformat(now_et()):
        raise ValueError("Admin permission expired")
    if not Path(root).resolve().is_relative_to(Path(payload["root_scope"]).resolve()):
        raise ValueError("Run is outside admin permission scope")
    if settings["mode"] not in payload["modes"] or settings["autonomy"] not in payload["autonomy"]:
        raise ValueError("Admin permission does not authorize this mode/autonomy")
    if action not in payload["actions"]:
        raise ValueError("Admin permission does not authorize " + action)
    return settings, payload


def renew_permission(root: Any, expected_permission: Any) -> Any:
    """A host admin session renews its existing scope after explicit approval."""
    settings, permission = permission_for(root, "approve", allow_expired=True)
    if permission != expected_permission:
        raise ValueError("Admin permission changed during approval")
    path = Path(settings["permission_file"])
    previous = path.read_bytes()
    archive = Path(root) / "admin/permission-history" / (hashlib.sha256(previous).hexdigest() + ".json")
    archive.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not archive.exists():
        descriptor = os.open(archive, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as file:
            file.write(previous)
    renewed = dict(permission)
    expires = datetime.fromisoformat(now_et()) + timedelta(hours=16)
    renewed["expires_et"] = expires.isoformat()
    write_json(path, {"permission": renewed, "signature": signature(renewed, admin_key())})
    return renewed


def candidate_state(candidate: Any) -> Any:
    state = dict(candidate)
    for field in ("grade", "status", "approval", "proposal", "timings_ms"):
        state.pop(field, None)
    return digest(state)


def signed_decision(root: Any, candidate: Any, grade: Any, *, explicit: Any = False, veto: Any = False) -> Any:
    if not veto and (candidate.get("admin_review", {}).get("hold") is True or "admin_veto" in candidate):
        raise ValueError("Candidate is held by the admin")
    settings, admin = permission_for(root, "approve" if explicit else "decide")
    if not veto:
        permission_for(root, "route")
    policy_hash = policy_identity(root, settings)
    payload = {"candidate_id": candidate["id"], "root": str(Path(root).resolve()),
               "state_sha256": candidate_state(candidate), "grade_sha256": digest(grade),
               "policy_sha256": policy_hash, "admin_id": admin["admin_id"],
               "mode": settings["mode"], "autonomy": settings["autonomy"],
               "explicit": explicit, "approved_et": now_et()}
    if veto:
        payload["admin_verdict"] = "reject"
    return {"decision": payload, "signature": signature(payload, admin_key())}


def apply_proposal(root: Any, candidate: Any, grade: Any) -> Any:
    record = dict(candidate)
    settings = settings_for(root)
    reason = "Manual autonomy requires admin approval"
    automatic = settings["autonomy"] == "automatic"
    if settings["autonomy"] == "supervised":
        automatic = grade["score"] >= settings["min_score"] and grade["route"] in settings["auto_routes"]
        if candidate["novelty"] == "conflicts" or candidate.get("change", {}).get("kind") == "update":
            automatic = False
        reason = "Proposal is outside supervised automatic approval criteria"
    if candidate.get("admin_review", {}).get("hold") is True or "admin_veto" in candidate:
        automatic = False
        reason = "Candidate is held by the admin"
    if automatic:
        try:
            record["approval"] = signed_decision(root, candidate, grade)
            record["grade"] = grade
            record["status"] = "accepted" if grade["verdict"] == "accept" else "rejected"
            record.pop("proposal", None)
            return record
        except (ValueError, OSError, KeyError) as error:
            reason = str(error)
    record["proposal"] = {"grade": grade, "reason": reason, "proposed_et": now_et()}
    record["grade"] = None
    record["status"] = "pending"
    return record


def approve(root: Any, identifiers: Any, *, expected_revisions: Any = None) -> Any:
    permission_for(root, "approve")
    records = latest_records(Path(root) / "queue/candidates.jsonl")
    selected = set(identifiers)
    available = {}
    for candidate in records:
        available[candidate["id"]] = candidate
    for identifier in selected:
        candidate = available.get(identifier)
        if candidate is None or candidate["status"] != "pending" or "proposal" not in candidate:
            raise ValueError("Approval requires existing pending GLM proposals")
        if candidate["proposal"]["grade"]["verdict"] == "accept" and candidate.get("review", {}).get("safe") is False:
            raise ValueError("Unsafe proposal cannot be accepted")
    approved = []
    replacements = []
    expected = {}
    for candidate in records:
        if candidate["id"] not in selected:
            continue
        if candidate["status"] != "pending" or "proposal" not in candidate:
            raise ValueError("Approval requires a pending GLM proposal")
        grade = candidate["proposal"]["grade"]
        if grade["verdict"] == "accept" and candidate.get("review", {}).get("safe") is False:
            raise ValueError("Unsafe proposal cannot be accepted")
        record = dict(candidate)
        record["approval"] = signed_decision(root, candidate, grade, explicit=True)
        record["grade"] = grade
        record["status"] = "accepted" if grade["verdict"] == "accept" else "rejected"
        record.pop("proposal", None)
        revision = record_revision(candidate)
        if expected_revisions is not None and expected_revisions.get(candidate["id"]) != revision:
            raise ValueError("Queue revision changed: " + candidate["id"])
        expected[candidate["id"]] = revision
        replacements.append(record)
        approved.append(candidate["id"])
    if set(approved) != selected:
        raise ValueError("Some requested candidate ids were absent")
    replace_records(Path(root) / "queue/candidates.jsonl", expected, replacements)
    return {"approved": approved}


def reject(root: Any, identifiers: Any, *, reason: Any, expected_revisions: Any) -> Any:
    """Admin veto preserves the GLM result and signs the separate rejection."""
    _, admin = permission_for(root, "approve")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Admin rejection requires a reason")
    selected = set(identifiers)
    expected = {}
    replacements = []
    for candidate in latest_records(Path(root) / "queue/candidates.jsonl"):
        if candidate["id"] not in selected:
            continue
        if candidate["status"] not in ("pending", "accepted"):
            raise ValueError("Admin rejection requires unfinished knowledge")
        revision = record_revision(candidate)
        if expected_revisions.get(candidate["id"]) != revision:
            raise ValueError("Queue revision changed: " + candidate["id"])
        record = dict(candidate)
        record["grade"] = candidate.get("grade") or candidate.get("proposal", {}).get("grade")
        record["status"] = "rejected"
        record["admin_veto"] = {"reason": reason, "admin_id": admin["admin_id"], "rejected_et": now_et()}
        record.pop("proposal", None)
        record["approval"] = signed_decision(root, record, record["grade"], explicit=True, veto=True)
        expected[candidate["id"]] = revision
        replacements.append(record)
    if set(expected) != selected:
        raise ValueError("Some requested candidate ids were absent")
    replace_records(Path(root) / "queue/candidates.jsonl", expected, replacements)
    return {"rejected": list(expected)}


def authorize_handoff(root: Any, candidate: Any) -> Any:
    if candidate.get("admin_review", {}).get("hold") is True:
        raise ValueError("Candidate is held by the admin")
    settings, _ = permission_for(root, "stage")
    approval = candidate.get("approval")
    if approval is None:
        raise ValueError("Accepted item requires an admin-approved decision receipt")
    payload = approval["decision"]
    if payload.get("admin_verdict") == "reject" or "admin_veto" in candidate:
        raise ValueError("Admin rejected this knowledge")
    if not hmac.compare_digest(approval["signature"], signature(payload, admin_key())):
        raise ValueError("Decision approval signature is invalid")
    if payload["root"] != str(Path(root).resolve()) or payload["candidate_id"] != candidate["id"]:
        raise ValueError("Approval belongs to a different run or candidate")
    if payload["state_sha256"] != candidate_state(candidate) or payload["grade_sha256"] != digest(candidate["grade"]):
        raise ValueError("Candidate or destination changed after admin approval")
    policy_hash = policy_identity(root, settings)
    if payload["policy_sha256"] != policy_hash:
        raise ValueError("Policy changed after admin approval")
