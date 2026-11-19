"""Guardrails.

The checks that stand between untrusted text and the user:

1. **Prompt-injection screening** -- on the user's question (direct injection) and
   on every retrieved passage (indirect injection: a poisoned document trying to
   give the model orders). A question above the threshold is refused before any
   tool runs; a passage above it is quarantined and never reaches the model.
2. **Schema validation** -- a tool call that does not match the contract never runs.
3. **Numeric grounding** -- every number in the final answer must trace back to a
   tool result. Used by the verification agent (see `verifier.py`).
4. **Quality gates** -- a degraded forecast or a peak above the client's trigger
   threshold flags the answer for human review rather than hiding the problem.

Personal-data redaction lives in `pii.py`; the SQL sandbox lives in `data.py`.

Injection screening here is pattern-based. It catches the common phrasings and
is cheap enough to run on every passage; it is one layer, not a guarantee. The
other layers are that retrieved text is fenced off as data in every prompt, and
that the tools themselves are sandboxed (read-only SQL, no file access), so a
model that *is* fooled still cannot do much.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from .config import settings
from .schemas import ToolCall

# --- prompt injection ---------------------------------------------------------

_INJECTION_RULES: list[tuple[str, float, re.Pattern[str]]] = [
    ("override_instructions", 1.0, re.compile(
        r"\b(ignore|disregard|forget|override|skip)\b[\w\s,]{0,30}?\b(previous|prior|above|earlier|preceding|all|your|the|system)\b"
        r"[\w\s]{0,20}?\b(instructions?|rules|prompts?|messages|guidelines|directions)\b", re.I)),
    ("reveal_system_prompt", 1.0, re.compile(
        r"\b(reveal|print|show|repeat|output|leak|display|dump|tell me)\b[\w\s]{0,20}?\b(system|hidden|initial|original|secret|developer)\b"
        r"[\w\s]{0,10}?\b(prompt|instructions?|message|rules)\b", re.I)),
    ("role_hijack", 0.6, re.compile(
        r"\b(you are now|from now on,? you|act as|pretend (to be|you are)|roleplay as|you must now)\b", re.I)),
    ("jailbreak_mode", 0.8, re.compile(r"\b(developer|dan|jailbreak|god|admin|sudo)\s+mode\b", re.I)),
    # Chat-template control tokens never appear in an honest question or document.
    ("chat_template_tokens", 1.0, re.compile(r"<\|(im_start|im_end|system|user|assistant)\|>|\[/?INST\]|<</?SYS>>", re.I)),
    ("fake_system_header", 0.8, re.compile(
        r"(</?system>|^\s*#{2,}\s*system\b|^\s*system\s*:|new instructions\s*:)", re.I | re.M)),
    ("bypass_safety", 0.8, re.compile(
        r"\b(bypass|disable|turn off|circumvent)\b[\w\s]{0,15}?\b(guardrails?|filters?|safety|rules|restrictions|checks?)\b", re.I)),
    ("hide_from_user", 0.6, re.compile(r"\b(do not|don't|never)\s+(tell|inform|show|mention to)\s+the\s+user\b", re.I)),
    ("exfiltration", 0.8, re.compile(
        r"\b(send|post|upload|forward|exfiltrate)\b[\w\s]{0,20}?\b(data|database|credentials|api key|secrets?|password)\b"
        r"[\w\s]{0,15}?\b(to|at)\b", re.I)),
]


@dataclass(frozen=True)
class InjectionVerdict:
    score: float
    rules: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.score >= settings.injection_block_score

    @property
    def suspicious(self) -> bool:
        return self.score > 0


def detect_prompt_injection(text: str) -> InjectionVerdict:
    """Score `text` for instruction-override attempts. Higher is worse."""
    hits = [(name, weight) for name, weight, pattern in _INJECTION_RULES if pattern.search(text or "")]
    score = round(min(2.0, sum(weight for _, weight in hits)), 2)
    return InjectionVerdict(score=score, rules=[name for name, _ in hits])


# --- tool calls ---------------------------------------------------------------

def validate_tool_call(raw: dict[str, Any]) -> tuple[ToolCall | None, str | None]:
    """Parse one proposed step. Returns (call, error)."""
    try:
        return ToolCall.model_validate(raw), None
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first.get("loc", []))
        return None, f"invalid tool call at '{location}': {first.get('msg')}"


# --- numeric grounding --------------------------------------------------------

NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
# Clock times ("17:30"), ISO dates and placeholder tokens are labels, not quantities.
LABEL_PATTERN = re.compile(r"\d{1,2}:\d{2}(:\d{2})?|\d{4}-\d{2}-\d{2}|\[[A-Z]+\]")
# Site identifiers such as NL-UTR-01 contain digits that are names, not numbers.
SITE_PATTERN = re.compile(r"\b[A-Z]{2}-[A-Z]{3}-\d{2}\b")

# Structural numbers that appear in prose ("24 h horizon", "one of 2 sites").
# Deliberately small: every extra entry is a number a model could invent for free.
ALWAYS_ALLOWED = {0.0, 1.0, 2.0, 24.0, 48.0, 100.0}


def extract_numbers(text: str) -> list[float]:
    text = SITE_PATTERN.sub(" ", LABEL_PATTERN.sub(" ", text or ""))
    found: list[float] = []
    for match in NUMBER_PATTERN.finditer(text):
        token = match.group(0).replace(",", "")
        if token in {"", "-", "."}:
            continue
        try:
            found.append(float(token))
        except ValueError:
            continue
    return found


# Only a time window the user typed may be echoed back ("over the last 30 days").
# Any other number in the question is a claim to check, not evidence: otherwise
# "Is the peak 812 kW?" would let an answer say "the peak is 812 kW" unchecked.
QUESTION_WINDOW = re.compile(r"\b(\d{1,4})\s*(?:-\s*)?(?:h|hrs?|hours?|days?|weeks?|months?|years?)\b", re.I)


def question_window_numbers(question: str) -> set[float]:
    return {float(m.group(1)) for m in QUESTION_WINDOW.finditer(question or "")}


def _allowed_values(evidence: list[dict[str, Any]], question: str = "") -> set[float]:
    allowed: set[float] = set(ALWAYS_ALLOWED)
    allowed.update(question_window_numbers(question))
    for item in evidence:
        for value in item.get("values", {}).values():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            allowed.update({
                number, round(number, 1), round(number, 2), float(round(number)),
                round(number * 100, 1), round(number * 100, 2),  # 0.084 -> 8.4 (%)
            })
        # Numbers quoted inside a tool summary or a retrieved passage are grounded
        # too: they came from the client's own data or documents.
        allowed.update(extract_numbers(item.get("summary", "")))
    return allowed


def check_numeric_grounding(
    answer: str,
    evidence: list[dict[str, Any]],
    question: str = "",
    tolerance: float = 0.005,
) -> list[float]:
    """Return the numbers in `answer` that no tool produced.

    The tolerance is relative (0.5%) so that "about 512 kW" passes against 512.4
    but a fabricated 19,500 does not pass against a measured 19,200.
    """
    allowed = _allowed_values(evidence, question)
    ungrounded: list[float] = []
    for number in extract_numbers(answer):
        if any(abs(number - c) <= max(0.051, abs(c) * tolerance) for c in allowed):
            continue
        ungrounded.append(number)
    return ungrounded


# --- quality gates ------------------------------------------------------------

def quality_flags(evidence: list[dict[str, Any]]) -> list[str]:
    """Operational gates that should reach the reader, not be swallowed."""
    flags: list[str] = []
    for item in evidence:
        values = item.get("values", {})
        mape = values.get("mape")
        if mape is not None and float(mape) > settings.mape_alert_threshold:
            flags.append("forecast_quality_below_threshold")
        used_pct = values.get("capacity_used_pct")
        if used_pct is not None and float(used_pct) >= 92.0:
            flags.append("peak_shaving_trigger_reached")
    doc_steps = [e for e in evidence if e.get("tool") in {"search_playbook", "summarize_documents"}]
    if doc_steps and not any(e.get("citations") for e in doc_steps):
        flags.append("no_supporting_documentation")
    return sorted(set(flags))
