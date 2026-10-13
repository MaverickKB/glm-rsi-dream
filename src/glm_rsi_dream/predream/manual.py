"""Prepared knowledge intake, with explicit provenance and no novelty probe."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..files import digest, invalid_constant, read_json, write_json
from .extract import discover_candidates, validate_candidates
from .queue import append_record, candidate_id, latest_records
from .timing import StageTimer, now_et


def secret_signal(text: Any) -> Any:
    """Block credential-shaped input before it reaches a trace or model."""
    patterns = [
        r"-----BEGIN (?:[A-Z ]*PRIVATE KEY|OPENSSH PRIVATE KEY)-----",
        r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,})\b",
        r"\bAKIA[A-Z0-9]{16}\b",
        r"(?i)\b(?:api[_ -]?key|access[_ -]?token|secret|password|bearer)\b[\s\"']*[:=][\s\"']*\S+",
    ]
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False


def prepared_rows(value: Any, *, learned: Any = False) -> Any:
    """A pack declares atomic facts rather than asking extraction to guess them."""
    if not isinstance(value, dict) or not isinstance(value.get("facts"), list) or not value["facts"]:
        raise ValueError("Prepared pack requires a nonempty facts array")
    if set(value) != {"facts"}:
        raise ValueError("Prepared pack only accepts facts")
    if len(value["facts"]) > 2000:
        raise ValueError("Prepared pack supports at most 2000 facts")
    rows = []
    statements = set()
    allowed = {"statement", "prompt_evidence"}
    if learned:
        allowed.update({"id", "receipt"})
    for fact in value["facts"]:
        if not isinstance(fact, dict):
            raise ValueError("Each prepared fact must be an object")
        if set(fact) - allowed:
            raise ValueError("Prepared fact contains unknown fields")
        statement = fact.get("statement")
        evidence = fact.get("prompt_evidence", statement)
        if not isinstance(statement, str) or not statement.strip():
            raise ValueError("Prepared fact requires a nonempty statement")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("Prepared evidence must be nonempty text")
        if len(statement) > 16000 or len(evidence) > 16000:
            raise ValueError("Prepared statement and evidence support at most 16000 characters")
        if not learned and statement in statements:
            raise ValueError("Prepared pack contains a duplicate statement")
        statements.add(statement)
        rows.append({"text": statement, "prompt_evidence": evidence})
    return rows


def prepared_secret_signal(raw: Any, input_format: Any) -> Any:
    if secret_signal(raw):
        return True
    if input_format == "json":
        value = json.loads(raw, parse_constant=invalid_constant)
        rows = prepared_rows(value)
        for row in rows:
            if secret_signal(row["text"]) or secret_signal(row["prompt_evidence"]):
                return True
    return False


def base_record(source_ref: Any, text: Any, evidence: Any) -> Any:
    return {"id": candidate_id(source_ref, text), "created_et": now_et(),
            "source": {"kind": "manual", "ref": source_ref}, "text": text,
            "prompt_evidence": evidence, "glm_prior": {"answer": "", "agrees": None},
            "novelty": "unmeasured", "status": "pending", "grade": None,
            "timings_ms": {"extract": 0, "probe": 0, "grade": 0, "absorb": 0}}


def import_learned(root: Any, path: Any) -> Any:
    """Register operator-supplied learned receipts, never infer learning from staging."""
    from .control import permission_for

    permission_for(root, "catalog")
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if secret_signal(raw):
        raise ValueError("Learned catalog contains credential-shaped material")
    value = json.loads(raw)
    rows = prepared_rows(value, learned=True)
    queue_path = Path(root) / "queue" / "candidates.jsonl"
    existing = {}
    for record in latest_records(queue_path):
        existing[record["id"]] = record
    imported = []
    seen_external = set()
    source_hash = hashlib.sha256(raw.encode()).hexdigest()
    for fact, row in zip(value["facts"], rows):
        external_id = fact.get("id")
        receipt = fact.get("receipt")
        if not isinstance(external_id, str) or not external_id.strip():
            raise ValueError("Learned fact requires an external id")
        if not isinstance(receipt, str) or not receipt.strip():
            raise ValueError("Learned fact requires a learning provenance receipt")
        if external_id in seen_external:
            raise ValueError("Learned catalog contains a duplicate external id")
        seen_external.add(external_id)
        for previous in existing.values():
            if previous.get("learned", {}).get("external_id") == external_id and previous["text"] != row["text"]:
                raise ValueError("Changed learned facts must enter as linked updates")
        record = base_record("learned:" + external_id, row["text"], row["prompt_evidence"])
        record["status"] = "learned"
        record["learned"] = {"external_id": external_id, "receipt": receipt,
                             "catalog": str(path), "catalog_sha256": source_hash}
        previous = existing.get(record["id"])
        if previous is not None:
            if previous["status"] != "learned" or previous.get("learned") != record["learned"]:
                raise ValueError("Learned catalog receipt changed; preserve the old catalog")
            continue
        imported.append(record)
    # Validate the entire catalog before appending any of it.
    for record in imported:
        append_record(queue_path, record)
    return imported


def learn_file(root: Any, path: Any, *, client: Any, source_ref: Any = None, policy: Any = None) -> Any:
    path = Path(path)
    source_ref = source_ref or str(path.resolve())
    raw = path.read_text(encoding="utf-8")
    input_format = "json" if path.suffix.lower() == ".json" else "text"
    return learn_content(root, raw, source_ref=source_ref, input_format=input_format,
                         client=client, policy=policy)


def learn_content(root: Any, raw: Any, *, source_ref: Any, input_format: Any, client: Any, policy: Any = None) -> Any:
    """File, API and portal intake share the same versioned queue identity."""
    root = Path(root)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Manual input must be nonempty")
    if not isinstance(source_ref, str) or not source_ref.strip():
        raise ValueError("Manual source reference must be nonempty")
    if input_format not in ("text", "json"):
        raise ValueError("Manual input format must be text or json")
    source_hash = hashlib.sha256(raw.encode()).hexdigest()
    blocked = prepared_secret_signal(raw, input_format)
    declared = None
    if input_format == "json" and not blocked:
        declared = prepared_rows(json.loads(raw, parse_constant=invalid_constant))
    receipt_id = hashlib.sha256((source_ref + source_hash).encode()).hexdigest()
    receipt_path = root / "intake" / "manual" / (receipt_id + ".json")
    with StageTimer(root, "manual_intake", source_ref) as timer:
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt.get("source_sha256") != source_hash:
                raise ValueError("Manual extraction receipt source changed")
            rows = receipt["rows"]
            if receipt.get("rows_sha256") != digest(rows):
                raise ValueError("Manual extraction receipt content changed")
            if not blocked and input_format == "text":
                validate_candidates({"candidates": rows}, raw)
            elif not blocked and rows != declared:
                raise ValueError("Prepared pack extraction receipt changed")
        elif blocked:
            rows = [{"text": "[REDACTED credential-bearing manual input]",
                     "prompt_evidence": "[REDACTED credential-bearing manual input]"}]
        elif input_format == "json":
            rows = declared
        else:
            rows = discover_candidates(raw, client=client, policy=policy)
        if not receipt_path.exists():
            write_json(receipt_path, {"source_sha256": source_hash, "rows": rows,
                                      "rows_sha256": digest(rows)})
    existing = set()
    recorded_version = set()
    previous_ids = {}
    queue_path = root / "queue" / "candidates.jsonl"
    for record in latest_records(queue_path):
        existing.add(record["id"])
        # Earlier sealed manual receipts already identify their file version.
        # Keep their original IDs on replay without rewriting approved content.
        same_ref = record["source"]["ref"] in (source_ref, source_ref + "#blocked-" + source_hash[:16])
        same_version = record.get("intake", {}).get("source_sha256") == source_hash
        if record["source"]["kind"] == "manual" and same_ref and same_version:
            recorded_version.add(record["text"])
            previous_ids[record["text"]] = record["id"]
    version_ref = source_ref + "#manual-" + source_hash
    queued = []
    identifiers = []
    for row in rows:
        record = base_record(version_ref, row["text"], row["prompt_evidence"])
        identifier = previous_ids.get(row["text"], record["id"])
        if identifier not in identifiers:
            identifiers.append(identifier)
        if record["id"] in existing or row["text"] in recorded_version:
            continue
        record["intake"] = {"mode": "prepared", "novelty_check": "skipped_manual",
                            "source_sha256": source_hash, "secret_preflight": "blocked" if blocked else "clear"}
        record["timings_ms"]["extract"] = timer.wall_ms / max(1, len(rows))
        append_record(queue_path, record)
        queued.append(record)
        existing.add(record["id"])
    return {"queued": queued, "candidate_ids": identifiers, "novelty_check": "skipped_manual",
            "secret_preflight": "blocked" if blocked else "clear"}
