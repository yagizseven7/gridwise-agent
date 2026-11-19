"""The verification agent.

A second agent, separate from the one that wrote the answer, checks the draft
before it is returned. It runs two independent checks and both must pass:

1. **Claim support** -- the draft is split into claims and each one is checked
   against the evidence units (tool summaries, retrieved passages, summary
   sentences). Live: an LLM judge with its own prompt. Offline: lexical overlap.
2. **Numeric grounding** -- every number in the draft must match a tool output
   (deterministic, see `guardrails.check_numeric_grounding`). A model judge is
   not trusted with arithmetic, so this check never depends on the LLM.

If the verifier's own output cannot be parsed it fails closed: the draft is
treated as unverified rather than waved through.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from . import guardrails
from .llm import LLM, LLMResponse, extract_json
from .schemas import ClaimCheck, Evidence, Verification


def evidence_units(evidence: list[Evidence]) -> list[dict[str, str]]:
    """Flatten evidence into small, individually citable units for checking."""
    units: list[dict[str, str]] = []
    for i, item in enumerate(evidence, start=1):
        if not item.ok:
            continue
        units.append({"id": f"E{i}", "text": item.brief or item.summary})
        if item.brief and item.summary != item.brief:
            units.append({"id": f"E{i}.summary", "text": item.summary})
        for j, passage in enumerate(item.raw.get("passages", []), start=1):
            units.append({"id": f"E{i}.p{j} {passage['citation']}", "text": passage["text"]})
        for j, sentence in enumerate(item.raw.get("sentences", []), start=1):
            units.append({"id": f"E{i}.s{j} {sentence['citation']}", "text": sentence["text"]})
    return units


class VerificationAgent:
    def __init__(self, llm: LLM):
        self.llm = llm

    def verify(self, question: str, answer: str, evidence: list[Evidence]) -> tuple[Verification, LLMResponse]:
        units = evidence_units(evidence)
        response = self.llm.verify(question, answer, units)

        claims: list[ClaimCheck] = []
        parse_problem: str | None = None
        try:
            payload: dict[str, Any] = extract_json(response.text)
            claims = [ClaimCheck.model_validate(c) for c in payload.get("claims", [])]
        except (ValueError, ValidationError) as exc:
            parse_problem = f"verifier output unreadable: {type(exc).__name__}"

        payload_dicts = [e.model_dump(exclude={"raw"}) for e in evidence]
        ungrounded = guardrails.check_numeric_grounding(answer, payload_dicts, question=question)
        unsupported = [c.claim for c in claims if not c.supported]
        if parse_problem:
            unsupported.append(parse_problem)

        verdict = Verification(
            grounded=not ungrounded and not unsupported,
            unsupported_claims=unsupported,
            ungrounded_numbers=ungrounded,
            claims=claims,
            verifier=response.provider,
        )
        return verdict, response
