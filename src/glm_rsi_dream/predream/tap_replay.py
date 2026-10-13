"""Strict offline replay of recorded GLM requests and responses, without sockets."""

import json
from pathlib import Path
from typing import Any

from .extract import COMPARE_INSTRUCTIONS, EXTRACT_INSTRUCTIONS, PRIOR_INSTRUCTIONS
from .tap import TrafficTap, drain


def stage(messages: Any) -> Any:
    system = messages[0]["content"]
    if system == EXTRACT_INSTRUCTIONS:
        return "extract"
    if system == PRIOR_INSTRUCTIONS:
        return "prior"
    if system.startswith(COMPARE_INSTRUCTIONS):
        return "compare"
    raise ValueError("Recorded replay requires a known GLM protocol")


class RecordedClient:
    """Match the entire non-system exchange and protocol, with no mock fallback."""

    def __init__(self, records: Any) -> None:
        self.answers: dict[Any, Any] = {}
        self.calls = 0
        for record in records:
            messages = record.get("messages", record.get("request", {}).get("messages"))
            kind = stage(messages)
            choice = record["response"]["choices"][0]
            if choice.get("finish_reason") == "length":
                continue
            answer = choice["message"]["content"]
            key = (kind, json.dumps(messages[1:], sort_keys=True))
            if key not in self.answers:
                self.answers[key] = answer

    def chat(self, messages: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise ValueError("Recorded traffic has no requested sampling variant")
        key = (stage(messages), json.dumps(messages[1:], sort_keys=True))
        if key not in self.answers:
            raise ValueError("No exact recorded exchange for this replay request")
        self.calls += 1
        return self.answers[key]


def replay(trace_path: Any, root: Any, *, policy: Any, principal: Any, opt_in: Any = False) -> Any:
    if not policy.enabled:
        return {"status": "disabled", "observed": {}, "processed": 0, "queued": 0, "model_calls": 0}
    if opt_in is not True or principal not in policy.principals:
        return {"status": "no_consent", "observed": {}, "processed": 0, "queued": 0, "model_calls": 0}
    records = []
    with Path(trace_path).open(encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    client = RecordedClient(records)
    tap = TrafficTap(root, policy)
    counts: dict[Any, Any] = {}
    for index, record in enumerate(records):
        messages = record.get("messages", record.get("request", {}).get("messages"))
        if stage(messages) != "extract":
            continue
        result = tap.observe({"messages": messages}, principal=principal,
                             request_id="recorded-" + str(index), opt_in=opt_in,
                             mode=policy.mode, endpoint=policy.endpoint)
        status = result["status"]
        counts[status] = counts.get(status, 0) + 1
    result = drain(root, policy=policy, client=client, max_events=policy.max_events)
    result.update({"observed": counts, "model_calls": client.calls,
                   "generation": "recorded GLM responses only; no live generation"})
    return result
