"""Prepared knowledge actions using the ordinary manual intake and GLM grader."""

from typing import Any

from .grade import grade_pending
from .manual import learn_content
from .queue import latest_records


def intake(root: Any, body: Any, *, client: Any, policy: Any) -> Any:
    result = learn_content(root, body["content"], source_ref=body["source_ref"],
                           input_format=body["format"], client=client, policy=policy.get("extract"))
    selected = []
    for record in latest_records(root / "queue/candidates.jsonl"):
        if record["id"] in result["candidate_ids"]:
            selected.append(record)
    queued_ids = []
    for record in result["queued"]:
        queued_ids.append(record["id"])
    return {"items": selected, "queued_ids": queued_ids,
            "novelty_check": result["novelty_check"], "secret_preflight": result["secret_preflight"]}


def grade(root: Any, candidate: Any, *, client: Any, policy: Any) -> Any:
    errors = []
    if candidate["status"] == "pending" and "proposal" not in candidate:
        result = grade_pending(root, client, candidate_id=candidate["id"], policy=policy.get("grade"))
        for error in result["errors"]:
            # Raw model/transport diagnostics stay in host-owned traces.
            errors.append({"id": error["id"], "error": "GLM review or grading failed. The fact remains pending; retry grading."})
    current = None
    for record in latest_records(root / "queue/candidates.jsonl"):
        if record["id"] == candidate["id"]:
            current = record
            break
    return {"item": current, "errors": errors}
