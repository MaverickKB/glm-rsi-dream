"""Closed request shapes shared by the local admin API and its OpenAPI document."""

import copy
import math
import re
from datetime import datetime
from typing import Any


def object_schema(properties: Any, required: Any = ()) -> Any:
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(required)}


NAME = {"type": "string", "pattern": "^[a-z0-9][a-z0-9_-]{0,63}$"}
TEXT = {"type": "string", "minLength": 1, "maxLength": 16000}
NOTE = {"type": "string", "maxLength": 2000}
REVISION = {"type": "integer", "minimum": 1}
STRINGS = {"type": "array", "items": TEXT, "maxItems": 200, "uniqueItems": True}
IDS = {"type": "array", "items": NAME, "minItems": 1, "maxItems": 200, "uniqueItems": True}
TIME = {"type": "string", "format": "date-time"}
FACT = object_schema({"id": NAME, "statement": TEXT,
                      "provenance": {"type": "object"}}, ("id", "statement", "provenance"))
FACTS = {"type": "array", "items": FACT, "maxItems": 2000}
CAPACITY = {"type": "integer", "minimum": 1, "maximum": 2000}
RULE_FIELDS: dict[str, Any] = {
    "name": {"type": "string", "minLength": 1, "maxLength": 100},
    "enabled": {"type": "boolean", "default": True},
    "domain": {"type": "string", "enum": ["general", "companion", "persona"]},
    "action": {"type": "string", "enum": ["propose", "ignore"]},
    "route": {"type": "string", "enum": ["weights", "pocket", "memory"]},
    "pocket": {"type": ["string", "null"], "pattern": NAME["pattern"], "default": None},
    "priority": {"type": "integer", "minimum": 0, "maximum": 1000, "default": 100},
    "user_ids": dict(STRINGS, default=[]),
    "exclude_user_ids": dict(STRINGS, default=[]),
    "source_kinds": {"type": "array", "items": {"type": "string", "enum": ["prompt", "conversation", "manual"]},
                     "minItems": 1, "uniqueItems": True, "default": ["prompt", "conversation", "manual"]},
    "patterns": dict(STRINGS, default=[]),
    "notes": dict(NOTE, default=""),
}
WINDOW = object_schema({
    "days": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 6},
             "minItems": 1, "maxItems": 7, "uniqueItems": True},
    "start": {"type": "string", "pattern": "^([01][0-9]|2[0-3]):[0-5][0-9]$"},
    "end": {"type": "string", "pattern": "^([01][0-9]|2[0-3]):[0-5][0-9]$"},
}, ("days", "start", "end"))
PACKAGE = object_schema({"schema_version": {"type": "integer", "enum": [1]},
    "kind": {"type": "string", "enum": ["raw_facts"]}, "id": NAME, "topic": NAME,
    "capacity": CAPACITY, "facts": FACTS,
    "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}},
    ("schema_version", "kind", "id", "topic", "capacity", "facts", "sha256"))
POLICY = object_schema({"schema_version": {"type": "integer", "enum": [1]}, "name": TEXT,
    "_comments": {"type": "object"},
    "extract": object_schema({
        "strategy": {"type": "string", "enum": ["closed_book", "claim_assessment"]},
        "probe_samples": {"type": "integer", "minimum": 1},
        "probe_temperature": {"type": ["number", "null"], "minimum": 0},
        "agreement_threshold": {"type": "number", "minimum": 0, "maximum": 1},
        "extraction_instructions": TEXT, "prior_instructions": TEXT,
        "comparison_instructions": TEXT, "assessment_instructions": TEXT}),
    "grade": object_schema({"strategy": {"type": "string", "enum": ["evidence_first", "custom"]},
        "instructions": {"type": ["string", "null"], "minLength": 1},
        "routing_rules": {"type": ["string", "null"], "minLength": 1},
        "max_attempts": {"type": "integer", "minimum": 1, "maximum": 3}}),
    "control": object_schema({"mode": {"type": "string", "enum": ["spark", "harness"]},
        "autonomy": {"type": "string", "enum": ["manual", "supervised", "automatic"]},
        "permission_file": {"type": ["string", "null"]},
        "auto_routes": {"type": "array", "items": {"type": "string", "enum": ["weights", "pocket", "memory"]}, "uniqueItems": True},
        "min_score": {"type": "number", "minimum": 0, "maximum": 1}})},
    ("schema_version", "name", "extract", "grade", "control"))

SCHEMAS = {
    "Learn": object_schema({"source_ref": {"type": "string", "minLength": 1, "maxLength": 512},
        "format": {"type": "string", "enum": ["text", "json"]},
        "content": {"type": "string", "minLength": 1, "maxLength": 1000000}},
        ("source_ref", "format", "content")),
    "RuleCreate": object_schema(dict(RULE_FIELDS, id=NAME), ("id", "name", "domain", "action", "route")),
    "RuleEdit": object_schema(dict(RULE_FIELDS, expected_revision=REVISION), ("expected_revision",)),
    "Delete": object_schema({"expected_revision": REVISION}, ("expected_revision",)),
    "Preset": object_schema({"id": NAME, "preset": {"type": "string", "enum": ["companion", "persona"]}}, ("id", "preset")),
    "RuleCheck": object_schema({"user_id": TEXT, "domain": {"type": "string", "enum": ["general", "companion", "persona"]},
                               "source_kind": {"type": "string", "enum": ["prompt", "conversation", "manual"]}, "text": TEXT},
                              ("user_id", "domain", "source_kind", "text")),
    "QueueEdit": object_schema({"expected_revision": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                                "hold": {"type": "boolean"}, "note": NOTE, "text": TEXT, "prompt_evidence": TEXT},
                               ("expected_revision",)),
    "Approve": object_schema({"expected_revision": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}, ("expected_revision",)),
    "Reject": object_schema({"expected_revision": {"type": "string", "pattern": "^[0-9a-f]{64}$"}, "reason": TEXT}, ("expected_revision", "reason")),
    "PolicyEdit": object_schema({"expected_revision": REVISION, "policy": POLICY}, ("expected_revision", "policy")),
    "Control": object_schema({"expected_revision": REVISION,
        "autonomy": {"type": "string", "enum": ["manual", "supervised", "automatic"]},
        "auto_routes": {"type": "array", "items": {"type": "string", "enum": ["weights", "pocket", "memory"]},
                        "uniqueItems": True},
        "min_score": {"type": "number", "minimum": 0, "maximum": 1}},
        ("expected_revision", "autonomy", "auto_routes", "min_score")),
    "Schedule": object_schema({"expected_revision": REVISION, "enabled": {"type": "boolean"},
        "timezone": {"type": "string", "minLength": 1, "maxLength": 80},
        "windows": {"type": "array", "items": WINDOW, "maxItems": 30},
        "min_idle_seconds": {"type": "integer", "minimum": 0, "maximum": 604800},
        "max_active_requests": {"type": "integer", "minimum": 0, "maximum": 1000}},
        ("expected_revision", "enabled", "timezone", "windows", "min_idle_seconds", "max_active_requests")),
    "DreamCreate": object_schema({"id": NAME, "run_id": NAME, "not_before": TIME}, ("id", "run_id", "not_before")),
    "DreamCheck": object_schema({"at": TIME, "idle_seconds": {"type": "integer", "minimum": 0},
                                 "active_requests": {"type": "integer", "minimum": 0}},
                                ("at", "idle_seconds", "active_requests")),
    "PocketCreate": object_schema({"id": NAME, "topic": NAME, "capacity": CAPACITY}, ("id", "topic", "capacity")),
    "PocketImport": object_schema({"id": NAME, "package": PACKAGE}, ("id", "package")),
    "PocketExtract": object_schema({"expected_revision": REVISION, "candidate_ids": IDS}, ("expected_revision", "candidate_ids")),
    "PocketCopy": object_schema({"id": NAME, "fact_ids": IDS}, ("id",)),
    "PocketMerge": object_schema({"id": NAME, "sources": dict(IDS, minItems=2), "capacity": CAPACITY}, ("id", "sources", "capacity")),
    "PocketRollover": object_schema({"expected_revision": REVISION, "id": NAME, "capacity": CAPACITY}, ("expected_revision", "id", "capacity")),
}


def reference(name: Any) -> Any:
    return {"$ref": "#/components/schemas/" + name}


def collection(name: Any) -> Any:
    return object_schema({"items": {"type": "array", "items": reference(name)}}, ("items",))


def response_schemas() -> Any:
    grade = object_schema({"verdict": {"type": "string", "enum": ["accept", "reject"]},
        "route": {"type": "string", "enum": ["weights", "pocket", "memory"]},
        "pocket": {"type": ["string", "null"]}, "reason": TEXT,
        "score": {"type": "number", "minimum": 0, "maximum": 1}, "grader": {"type": "string", "const": "glm"}},
        ("verdict", "route", "pocket", "reason", "score", "grader"))
    queue = {"type": "object", "required": ["id", "text", "source", "novelty", "status", "grade", "revision"],
        "properties": {"id": NAME, "text": TEXT, "prompt_evidence": TEXT, "created_et": TIME,
            "revision": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "source": object_schema({"kind": {"type": "string", "enum": ["prompt", "conversation", "manual"]}, "ref": TEXT}, ("kind", "ref")),
            "novelty": {"type": "string", "enum": ["new", "known", "conflicts", "unmeasured"]},
            "status": {"type": "string", "enum": ["pending", "accepted", "rejected", "staged", "absorbed", "learned"]},
            "grade": {"anyOf": [reference("Grade"), {"type": "null"}]},
            "proposal": object_schema({"grade": reference("Grade"), "reason": TEXT, "proposed_et": TIME}, ("grade", "reason", "proposed_et")),
            "admin_review": {"type": "object", "properties": {"hold": {"type": "boolean"}, "note": NOTE, "replacement_id": NAME}},
            "review": {"type": "object"}, "change": {"type": "object"}, "approval": {"type": "object"},
            "admin_veto": object_schema({"reason": TEXT, "admin_id": TEXT, "rejected_et": TIME}, ("reason", "admin_id", "rejected_et")),
            "glm_prior": {"type": "object"}, "timings_ms": {"type": "object"}}}
    rule = object_schema(dict(RULE_FIELDS, id=NAME, revision=REVISION), tuple(RULE_FIELDS) + ("id", "revision"))
    schedule_fields = dict(SCHEMAS["Schedule"]["properties"])
    schedule_fields.pop("expected_revision")
    schedule = object_schema(dict(schedule_fields, revision=REVISION), tuple(schedule_fields) + ("revision",))
    control_fields = dict(SCHEMAS["Control"]["properties"])
    control_fields.pop("expected_revision")
    control_fields["mode"] = {"type": "string", "enum": ["spark", "harness"]}
    control_view = object_schema(dict(control_fields, revision=REVISION), tuple(control_fields) + ("revision",))
    pocket = object_schema({"id": NAME, "topic": NAME, "capacity": CAPACITY, "facts": FACTS,
        "revision": REVISION, "sealed": {"type": "boolean"}, "shared": {"type": "boolean"},
        "kind": {"type": "string", "const": "raw_facts"}, "origin": {"type": "object"},
        "successor": {"type": ["string", "null"]}},
        ("id", "topic", "capacity", "facts", "revision", "sealed", "shared", "kind", "origin", "successor"))
    job = object_schema({"id": NAME, "run_id": NAME, "not_before": TIME, "revision": REVISION,
        "status": {"type": "string", "enum": ["scheduled", "cancelled"]}, "approved_by": TEXT, "approved_et": TIME},
        ("id", "run_id", "not_before", "revision", "status", "approved_by", "approved_et"))
    return {
        "LearnResult": object_schema({"items": {"type": "array", "items": reference("QueueItem")},
            "queued_ids": {"type": "array", "items": NAME},
            "novelty_check": {"type": "string", "const": "skipped_manual"},
            "secret_preflight": {"type": "string", "const": "clear"}},
            ("items", "queued_ids", "novelty_check", "secret_preflight")),
        "GradeResult": object_schema({"item": reference("QueueItem"),
            "errors": {"type": "array", "items": object_schema({"id": NAME, "error": TEXT}, ("id", "error"))}},
            ("item", "errors")),
        "Grade": grade, "QueueItem": queue, "QueueList": collection("QueueItem"),
        "QueueEditResult": object_schema({"item": reference("QueueItem"), "replacement_id": {"type": ["string", "null"]}}, ("item", "replacement_id")),
        "Rule": rule, "RuleList": collection("Rule"),
        "PresetList": object_schema({"items": {"type": "array", "items": {"type": "object"}}}, ("items",)),
        "RuleCheckResult": object_schema({"matches": {"type": "array", "items": reference("Rule")},
            "selected": {"anyOf": [reference("Rule"), {"type": "null"}]},
            "effect": {"type": "string", "const": "proposal_policy_only"}, "collection_enabled": {"type": "boolean", "const": False}},
            ("matches", "selected", "effect", "collection_enabled")),
        "ControlView": control_view, "ScheduleView": schedule,
        "PolicyView": object_schema({"revision": REVISION, "policy": POLICY,
            "scope": {"type": "string", "const": "next_workspace"},
            "active_policy": {"type": ["object", "null"]}}, ("revision", "policy", "scope", "active_policy")),
        "Pocket": pocket, "PocketList": collection("Pocket"), "PocketPackage": PACKAGE,
        "Rollover": object_schema({"previous": reference("Pocket"), "successor": reference("Pocket")}, ("previous", "successor")),
        "DreamJob": job, "DreamList": collection("DreamJob"),
        "DreamReadiness": object_schema({"id": NAME, "eligible": {"type": "boolean"}, "reasons": STRINGS,
            "context": SCHEMAS["DreamCheck"], "context_source": {"type": "string", "const": "admin_supplied"},
            "next_action": {"type": "string", "enum": ["human_confirm", "wait"]},
            "plan": {"type": ["object", "null"]}, "executes_training": {"type": "boolean", "const": False}},
            ("id", "eligible", "reasons", "context", "context_source", "next_action", "plan", "executes_training")),
        "Removed": object_schema({"removed": NAME}, ("removed",)),
        "System": object_schema({"control": reference("ControlView"), "authority": {"type": "object"},
            "collection_enabled": {"type": "boolean", "const": False}, "training_execution": {"type": "boolean", "const": False},
            "pocket_kind": {"type": "string", "const": "raw_facts"},
            "features": object_schema({"dreams": object_schema({"enabled": {"type": "boolean"},
                "label": {"type": "string", "enum": ["Under development", "Manual preparation"]},
                "status": {"type": "string", "enum": ["Under development", "Manual preparation"]},
                "owner": {"type": "string", "const": "dream-neurons"},
                "opening_criterion": {"type": "string", "const": "Pocket-OFF recall bar passes"}},
                ("enabled", "label", "status", "owner", "opening_criterion"))}, ("dreams",))},
            ("control", "authority", "collection_enabled", "training_execution", "pocket_kind", "features")),
    }


RESPONSE_NAMES = {
    "learn_create": "LearnResult", "queue_grade": "GradeResult",
    "system": "System", "control_get": "ControlView", "control_put": "ControlView", "control_approve": "ControlView",
    "rules_list": "RuleList", "rules_create": "Rule", "rules_get": "Rule", "rules_edit": "Rule", "rules_preset": "Rule",
    "rules_delete": "Removed", "rules_presets": "PresetList", "rules_check": "RuleCheckResult",
    "queue_list": "QueueList", "queue_get": "QueueItem", "queue_edit": "QueueEditResult", "queue_approve": "QueueItem",
    "queue_reject": "QueueItem", "policy_get": "PolicyView", "policy_put": "PolicyView",
    "schedule_get": "ScheduleView", "schedule_put": "ScheduleView",
    "dreams_list": "DreamList", "dreams_create": "DreamJob", "dreams_get": "DreamJob", "dreams_cancel": "DreamJob", "dreams_check": "DreamReadiness",
    "pockets_list": "PocketList", "pockets_create": "Pocket", "pockets_get": "Pocket", "pockets_delete": "Removed",
    "pockets_import": "Pocket", "pockets_merge": "Pocket", "pockets_extract": "Pocket", "pockets_copy": "Pocket", "pockets_rollover": "Rollover",
    "pockets_export": "PocketPackage",
}

# method, path, operation, request schema, success status
OPERATIONS = [
    ("post", "/learn", "learn_create", "Learn", 200),
    ("post", "/queue/{id}/grade", "queue_grade", "Approve", 200),
    ("get", "/system", "system", None, 200), ("get", "/control", "control_get", None, 200),
    ("put", "/control", "control_put", "Control", 200),
    ("post", "/control/approve", "control_approve", "Control", 200),
    ("get", "/policy", "policy_get", None, 200), ("put", "/policy", "policy_put", "PolicyEdit", 200),
    ("get", "/rules", "rules_list", None, 200), ("post", "/rules", "rules_create", "RuleCreate", 201),
    ("get", "/rules/presets", "rules_presets", None, 200), ("post", "/rules/presets", "rules_preset", "Preset", 201),
    ("post", "/rules/check", "rules_check", "RuleCheck", 200),
    ("get", "/rules/{id}", "rules_get", None, 200), ("patch", "/rules/{id}", "rules_edit", "RuleEdit", 200),
    ("delete", "/rules/{id}", "rules_delete", "Delete", 200),
    ("get", "/queue", "queue_list", None, 200), ("get", "/queue/{id}", "queue_get", None, 200),
    ("patch", "/queue/{id}", "queue_edit", "QueueEdit", 200),
    ("post", "/queue/{id}/approve", "queue_approve", "Approve", 200),
    ("post", "/queue/{id}/reject", "queue_reject", "Reject", 200),
    ("get", "/schedule", "schedule_get", None, 200), ("put", "/schedule", "schedule_put", "Schedule", 200),
    ("get", "/dreams", "dreams_list", None, 200), ("post", "/dreams", "dreams_create", "DreamCreate", 201),
    ("get", "/dreams/{id}", "dreams_get", None, 200), ("delete", "/dreams/{id}", "dreams_cancel", "Delete", 200),
    ("post", "/dreams/{id}/check", "dreams_check", "DreamCheck", 200),
    ("get", "/pockets", "pockets_list", None, 200), ("post", "/pockets", "pockets_create", "PocketCreate", 201),
    ("post", "/pockets/import", "pockets_import", "PocketImport", 201),
    ("post", "/pockets/merge", "pockets_merge", "PocketMerge", 201),
    ("get", "/pockets/{id}", "pockets_get", None, 200), ("delete", "/pockets/{id}", "pockets_delete", "Delete", 200),
    ("get", "/pockets/{id}/export", "pockets_export", None, 200),
    ("post", "/pockets/{id}/extract", "pockets_extract", "PocketExtract", 200),
    ("post", "/pockets/{id}/copy", "pockets_copy", "PocketCopy", 201),
    ("post", "/pockets/{id}/rollover", "pockets_rollover", "PocketRollover", 201),
]


def validate(value: Any, schema: Any, location: Any = "body") -> Any:
    expected = schema.get("type")
    kinds: dict[str, Any] = {"object": dict, "array": list, "string": str, "integer": int,
             "number": (int, float), "boolean": bool, "null": type(None)}
    allowed = expected if isinstance(expected, list) else [expected]
    matches = False
    for kind in allowed:
        if type(value) in (kinds[kind] if isinstance(kinds[kind], tuple) else (kinds[kind],)):
            matches = True
    if not matches:
        raise ValueError(location + " has an invalid type")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(location + " has an invalid value")
    if value is None:
        return
    if isinstance(value, dict):
        fields = schema.get("properties", {})
        if schema.get("additionalProperties") is False and set(value) - set(fields):
            raise ValueError(location + " contains unknown fields")
        for name in schema.get("required", []):
            if name not in value:
                raise ValueError(location + " requires " + name)
        for name, field in fields.items():
            if name in value:
                validate(value[name], field, location + "." + name)
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", 10000):
            raise ValueError(location + " has an invalid item count")
        seen = []
        for item in value:
            validate(item, schema["items"], location + "[]")
            if schema.get("uniqueItems") and item in seen:
                raise ValueError(location + " has duplicate items")
            seen.append(item)
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", 1000000):
            raise ValueError(location + " has an invalid length")
        if schema.get("minLength", 0) > 0 and not value.strip():
            raise ValueError(location + " must contain text")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            raise ValueError(location + " has an invalid format")
        if schema.get("format") == "date-time":
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})", value) is None:
                raise ValueError(location + " requires an RFC3339 timestamp")
            if datetime.fromisoformat(value).tzinfo is None:
                raise ValueError(location + " requires a timezone")
    if type(value) in (int, float):
        if not math.isfinite(value):
            raise ValueError(location + " must be finite")
        if value < schema.get("minimum", value) or value > schema.get("maximum", value):
            raise ValueError(location + " is outside the allowed range")


def openapi_spec() -> Any:
    paths: dict[Any, Any] = {}
    for method, route, operation, request, status in OPERATIONS:
        entry: dict[str, Any] = {"operationId": operation, "tags": [route.split("/")[1]],
                 "responses": {str(status): {"description": "Successful operation", "content": {
                     "application/json": {"schema": reference(RESPONSE_NAMES[operation])}}}}}
        if operation.startswith(("dreams_", "schedule_")):
            entry["x-feature-gate"] = "dream_features_enabled"
            entry["description"] = ("Under development. Default server flag is off; closed gate returns "
                                    "503 feature_unavailable: under development. Owner opens after the "
                                    "pocket-OFF recall bar passes. Enabled actions prepare manual plans only.")
        for code in (400, 401, 403, 404, 409, 413, 415, 503):
            entry["responses"][str(code)] = {"description": "Request rejected", "content": {
                "application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}}
        if "{id}" in route:
            entry["parameters"] = [{"name": "id", "in": "path", "required": True, "schema": NAME}]
        if request is not None:
            entry["requestBody"] = {"required": True, "content": {
                "application/json": {"schema": {"$ref": "#/components/schemas/" + request}}}}
        paths.setdefault("/api/v1" + route, {})[method] = entry
    schemas = copy.deepcopy(SCHEMAS)
    schemas.update(response_schemas())
    schemas["Error"] = object_schema({"error": TEXT}, ("error",))
    return {"openapi": "3.1.0", "info": {"title": "RSI Dreambox API", "version": "0.1.0",
            "description": "Local admin control for portable LOOP2. Pocket packages contain raw facts. Dream and schedule features are Under development behind a default-off server flag. The owner opens it after the pocket-OFF recall bar passes; enabled checks return manual preparation plans, never training execution."},
            "servers": [{"url": "/"}], "security": [{"AdminBearer": []}], "paths": paths,
            "components": {"securitySchemes": {"AdminBearer": {"type": "http", "scheme": "bearer"}}, "schemas": schemas}}
