"""Explicit capture consent, conservative privacy checks and bounded defaults."""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class TapBudgetError(ValueError):
    """A content-free limit code that the operator can act on."""


@dataclass
class TapPolicy:
    enabled: bool = False
    mode: str = "harness"
    endpoint: str = ""
    principals: list[str] = field(default_factory=list)
    denied_terms: list[str] = field(default_factory=list)
    max_text_bytes: int = 8192
    max_events: int = 256
    max_spool_bytes: int = 2_000_000
    retention_seconds: int = 86400
    events_per_minute: int = 12
    principal_events_per_minute: int = 4
    model_calls_per_minute: int = 24
    calls_per_event: int = 24
    candidates_per_event: int = 8
    pending_candidates: int = 256

    def __post_init__(self) -> Any:
        if type(self.enabled) is not bool or self.mode not in {"spark", "harness"}:
            raise ValueError("Tap requires boolean enabled and spark/harness mode")
        if not isinstance(self.endpoint, str):
            raise ValueError("endpoint must be text")
        if self.enabled:
            address = urlsplit(self.endpoint)
            if address.scheme not in {"http", "https"} or address.port != 8888 or not address.hostname:
                raise ValueError("Enabled tap requires its exact head endpoint on8888")
            if address.username is not None or address.password is not None or address.query or address.fragment:
                raise ValueError("Tap endpoint must not include credentials, query or fragment")
        for name in ("principals", "denied_terms"):
            values = getattr(self, name)
            if not isinstance(values, list):
                raise ValueError(name + " must be a list")
            for value in values:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(name + " requires nonempty strings")
        for name, value in asdict(self).items():
            if name in {"enabled", "mode", "endpoint", "principals", "denied_terms"}:
                continue
            if type(value) is not int or value <= 0:
                raise ValueError(name + " must be a positive integer")

        if self.retention_seconds < 60:
            raise ValueError("retention_seconds must cover the sixty-second rate window")

    @classmethod
    def load(cls, path: Any) -> Any:
        if path is None:
            return cls()
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Tap policy must be an object")
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError("Unknown tap policy field") from error


# Reject the whole event rather than store a damaged fact beside a redaction.
PRIVATE_PATTERNS = [
    r"\b(?:password|passwd|passphrase|secret|credential|api[_ -]?key|access[_ -]?token|refresh[_ -]?token|bearer|authorization|cookie|session[_ -]?id|private[_ -]?key)\b",
    r"\b(?:sk|hf|ghp|github_pat|xai)[-_][A-Za-z0-9_-]{12,}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    r"(?<![A-Za-z0-9])(?:\+?\d[\d ().-]{7,}\d)(?![A-Za-z0-9])",
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"\b(?:ssn|social security|credit card|bank account|routing number|medical|diagnosis|prescription|patient|passport)\b",
    r"\b\d+\s+[^\n,]{1,60}\s(?:street|avenue|road|lane|drive|boulevard)\b",
    r"\b(?:do not|don't|never)\s+(?:save|store|remember|retain|learn|record)\b",
]


def privacy_reason(text: Any, policy: Any) -> Any:
    if not isinstance(text, str) or not text.strip():
        return "empty"
    if len(text.encode("utf-8")) > policy.max_text_bytes:
        return "oversize"
    for char in text:
        if ord(char) < 32 and char not in "\n\r\t":
            return "control_character"
    for term in policy.denied_terms:
        if term.casefold() in text.casefold():
            return "denied_term"
    for pattern in PRIVATE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return "private_content"
    return None
