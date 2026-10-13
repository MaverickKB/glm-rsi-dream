"""Portable raw-fact pocket operations. No tensor loading or serving mutation."""

import copy
from typing import Any

from . import control
from .absorb import approved_route
from .admin_schema import PACKAGE, validate
from .manual import secret_signal
from .review import verify_targets


def require_revision(item: Any, expected: Any) -> Any:
    if item["revision"] != expected:
        raise ValueError("Revision changed")


def check_facts(facts: Any, capacity: Any) -> Any:
    if len(facts) > capacity:
        raise ValueError("Pocket capacity exceeded")
    seen = set()
    for fact in facts:
        if fact["id"] in seen:
            raise ValueError("Duplicate pocket fact id")
        if secret_signal(fact["statement"]):
            raise ValueError("Pocket contains credential-shaped input")
        seen.add(fact["id"])


def new_pocket(identifier: Any, topic: Any, capacity: Any, *, facts: Any = None, origin: Any = None) -> Any:
    facts = copy.deepcopy(facts or [])
    check_facts(facts, capacity)
    return {"id": identifier, "topic": topic, "capacity": capacity, "facts": facts,
            "revision": 1, "sealed": False, "shared": False, "kind": "raw_facts",
            "origin": origin or {"operation": "create"}, "successor": None}


def export_pocket(pocket: Any) -> Any:
    package = {"schema_version": 1, "kind": "raw_facts", "id": pocket["id"],
               "topic": pocket["topic"], "capacity": pocket["capacity"],
               "facts": copy.deepcopy(pocket["facts"])}
    package["sha256"] = control.digest(package)
    return package


def import_pocket(identifier: Any, package: Any) -> Any:
    validate(package, PACKAGE)
    unsigned = dict(package)
    checksum = unsigned.pop("sha256")
    if checksum != control.digest(unsigned):
        raise ValueError("Shared pocket checksum does not match")
    pocket = new_pocket(identifier, package["topic"], package["capacity"], facts=package["facts"],
                        origin={"operation": "import", "source": package["id"], "sha256": checksum})
    pocket["shared"] = True
    return pocket


def copy_pocket(source: Any, identifier: Any, fact_ids: Any = None) -> Any:
    facts = []
    selected = set(fact_ids) if fact_ids is not None else None
    found = set()
    for fact in source["facts"]:
        if selected is None or fact["id"] in selected:
            facts.append(fact)
            found.add(fact["id"])
    if selected is not None and found != selected:
        raise ValueError("Copy requested an unknown fact")
    return new_pocket(identifier, source["topic"], source["capacity"], facts=facts,
                      origin={"operation": "copy", "source": source["id"],
                              "sha256": export_pocket(source)["sha256"]})


def merge_pockets(identifier: Any, sources: Any, capacity: Any) -> Any:
    topic = sources[0]["topic"]
    facts = []
    by_id: dict[Any, Any] = {}
    origins = []
    for source in sources:
        if source["topic"] != topic:
            raise ValueError("Merge requires one declared topic")
        origins.append({"id": source["id"], "sha256": export_pocket(source)["sha256"]})
        for fact in source["facts"]:
            existing = by_id.get(fact["id"])
            if existing is not None:
                if existing != fact:
                    raise ValueError("Merge has conflicting fact ids")
                continue
            by_id[fact["id"]] = fact
            facts.append(fact)
    return new_pocket(identifier, topic, capacity, facts=facts,
                      origin={"operation": "merge", "sources": origins})


def extract_facts(root: Any, pocket: Any, candidates: Any, expected_revision: Any) -> Any:
    require_revision(pocket, expected_revision)
    if pocket["sealed"]:
        raise ValueError("Pocket is sealed")
    facts = copy.deepcopy(pocket["facts"])
    known = {}
    for fact in facts:
        known[fact["id"]] = fact
    for candidate in candidates:
        if candidate["status"] not in ("accepted", "staged") or approved_route(candidate) == "memory":
            raise ValueError("Extract requires approved reusable knowledge")
        control.authorize_handoff(root, candidate)
        verify_targets(root, candidate)
        fact = {"id": candidate["id"], "statement": candidate["text"],
                "provenance": {"source": candidate["source"],
                               "approval_sha256": control.digest(candidate["approval"])}}
        if "change" in candidate:
            fact["provenance"]["change"] = copy.deepcopy(candidate["change"])
        existing = known.get(fact["id"])
        if existing is not None:
            if existing != fact:
                raise ValueError("Existing pocket fact has different content")
            continue
        facts.append(fact)
        known[fact["id"]] = fact
    check_facts(facts, pocket["capacity"])
    updated = copy.deepcopy(pocket)
    updated["facts"] = facts
    updated["revision"] += 1
    return updated


def rollover_pocket(source: Any, identifier: Any, capacity: Any, expected_revision: Any) -> Any:
    require_revision(source, expected_revision)
    if source["sealed"] or len(source["facts"]) < source["capacity"]:
        raise ValueError("Rollover requires a full open pocket")
    successor = new_pocket(identifier, source["topic"], capacity,
                           origin={"operation": "rollover", "source": source["id"],
                                   "sha256": export_pocket(source)["sha256"]})
    previous = copy.deepcopy(source)
    previous["sealed"] = True
    previous["successor"] = identifier
    previous["revision"] += 1
    return previous, successor
