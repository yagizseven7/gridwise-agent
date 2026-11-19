"""Personal data detection and redaction.

Personal data is removed from the question *before* it reaches a language model
or the trace log, and from the answer before it reaches the user. Detection is
pattern-based with checksums where the format has one (IBAN mod-97, BSN 11-proef,
card numbers Luhn), which keeps false positives on ordinary numbers low.

This is a rule-based filter sized for a personal project, not a DLP product: it
finds the structured identifiers below, not names or free-text descriptions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,3})?\b")
CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
BSN = re.compile(r"\b\d{3}[.\s]?\d{2}[.\s]?\d{2}[.\s]?\d{2}\b|\b\d{9}\b")
PHONE = re.compile(
    r"(?:(?<!\w)(?:\+|00)\d{2}[\s-]?(?:\(0\))?[\s-]?[1-9](?:[\s-]?\d){7,9}(?!\d))"
    r"|(?:\b0[1-9](?:[\s-]?\d){8}\b)"
)


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def iban_valid(candidate: str) -> bool:
    compact = candidate.replace(" ", "").upper()
    if not 15 <= len(compact) <= 34:
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


def bsn_valid(candidate: str) -> bool:
    """Dutch citizen service number: the '11-proef' checksum."""
    digits = _digits(candidate)
    if len(digits) != 9 or digits == "000000000":
        return False
    weights = [9, 8, 7, 6, 5, 4, 3, 2, -1]
    return sum(int(d) * w for d, w in zip(digits, weights)) % 11 == 0


def luhn_valid(candidate: str) -> bool:
    digits = _digits(candidate)
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


@dataclass(frozen=True)
class PIIMatch:
    kind: str
    start: int
    end: int
    text: str


# Order matters: longer, checksummed formats first so their digits are not
# re-claimed by a looser pattern.
_DETECTORS: list[tuple[str, re.Pattern[str], Callable[[str], bool]]] = [
    ("EMAIL", EMAIL, lambda _: True),
    ("IBAN", IBAN, iban_valid),
    ("CARD", CARD, luhn_valid),
    ("BSN", BSN, bsn_valid),
    ("PHONE", PHONE, lambda _: True),
]


def find_pii(text: str) -> list[PIIMatch]:
    found: list[PIIMatch] = []
    taken: list[tuple[int, int]] = []
    for kind, pattern, is_valid in _DETECTORS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < t_end and end > t_start for t_start, t_end in taken):
                continue
            if not is_valid(match.group(0)):
                continue
            found.append(PIIMatch(kind, start, end, match.group(0)))
            taken.append((start, end))
    return sorted(found, key=lambda m: m.start)


def redact(text: str) -> tuple[str, list[str]]:
    """Replace each identifier with a typed placeholder such as [EMAIL]."""
    matches = find_pii(text)
    if not matches:
        return text, []
    pieces: list[str] = []
    cursor = 0
    for m in matches:
        pieces.append(text[cursor:m.start])
        pieces.append(f"[{m.kind}]")
        cursor = m.end
    pieces.append(text[cursor:])
    return "".join(pieces), sorted({m.kind for m in matches})


def redact_obj(value: Any) -> Any:
    """Redact every string inside a nested dict/list structure (for logs)."""
    if isinstance(value, str):
        return redact(value)[0]
    if isinstance(value, dict):
        return {k: redact_obj(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_obj(v) for v in value]
    return value
