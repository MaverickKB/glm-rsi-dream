"""Memory, swap and exact reply preservation gates."""
import math


class SafetyError(RuntimeError):
    pass


class MemoryGate:
    def __init__(self, policy):
        self.policy = policy
        self.previous = {}
        self.high_since = {}
        self.minima = {}

    def observe(self, rank, sample):
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


def response_bytes(response, require_complete=True):
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


def require_parity(before, after, require_complete=True):
    if len(before) != len(after) or not before:
        raise SafetyError("parity probe set is empty or incomplete")
    for original, candidate in zip(before, after):
        if response_bytes(original, require_complete) != response_bytes(candidate, require_complete):
            raise SafetyError("generated reply byte parity failed")
