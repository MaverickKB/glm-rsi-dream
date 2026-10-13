"""Memory, swap and exact reply preservation gates."""

import math
from typing import Any


class SafetyError(RuntimeError):
    pass


class MemoryGate:
    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.previous: dict[Any, Any] = {}
        self.high_since: dict[Any, Any] = {}
        self.minima: dict[Any, Any] = {}

    def observe(self, rank: Any, sample: Any) -> Any:
        for key in ("available", "time", "swap_out_bytes"):
            value = sample.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise SafetyError("memory telemetry is invalid")
        available = sample["available"]
        self.minima[rank] = min(available, self.minima.get(rank, available))
        if available < self.policy["memory_floor_bytes"]:
            raise SafetyError("memory floor failed on rank " + str(rank))
        previous = self.previous.get(rank)
        if previous is not None:
            elapsed = sample["time"] - previous["time"]
            if elapsed <= 0 or sample["swap_out_bytes"] < previous["swap_out_bytes"]:
                raise SafetyError("memory telemetry is invalid")
            rate = (sample["swap_out_bytes"] - previous["swap_out_bytes"]) / elapsed
            if rate > self.policy["swap_out_bytes_per_second"]:
                self.high_since.setdefault(rank, previous["time"])
                if sample["time"] - self.high_since[rank] >= self.policy["swap_sustain_seconds"]:
                    raise SafetyError("sustained swap-out gate failed on rank " + str(rank))
            else:
                self.high_since.pop(rank, None)
        self.previous[rank] = sample


def response_bytes(response: Any, require_complete: Any = True) -> Any:
    """Compare generated UTF-8 bytes, token counts and finish, excluding request IDs."""
    choice = response["choices"][0]
    message = choice.get("message")
    if message is None:
        content = choice["text"]
    else:
        content = message.get("content") or ""
    if not isinstance(content, str):
        raise SafetyError("generated content is not text")
    if not content.strip() or (require_complete and choice.get("finish_reason") != "stop"):
        raise SafetyError("preservation reply is empty or incomplete")
    return content.encode("utf-8"), choice.get("finish_reason"), response.get("usage", {}).get("completion_tokens")


def require_parity(before: Any, after: Any, require_complete: Any = True) -> Any:
    if len(before) != len(after) or not before:
        raise SafetyError("parity probe set is empty or incomplete")
    for original, candidate in zip(before, after):
        if response_bytes(original, require_complete) != response_bytes(candidate, require_complete):
            raise SafetyError("generated reply byte parity failed")
