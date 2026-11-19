"""Typed contracts between the models and the rest of the system.

The models return text. Text is not a contract. Every time a model is allowed to
influence control flow, its output is parsed into one of these Pydantic models
first. If it does not validate, it does not run.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ToolName = Literal[
    "query_sensor_data",
    "forecast_demand",
    "search_playbook",
    "summarize_documents",
    "estimate_imbalance_cost",
]

SiteId = Literal["NL-UTR-01", "NL-AMS-02"]


class ToolCall(BaseModel):
    """One action the router wants to take."""

    tool: ToolName
    args: dict[str, Any] = Field(default_factory=dict)
    why: str = Field(default="", max_length=300)


class Plan(BaseModel):
    """The router agent's decision: an ordered list of tool calls."""

    objective: str = Field(max_length=400)
    steps: list[ToolCall] = Field(min_length=1, max_length=6)

    @field_validator("steps")
    @classmethod
    def _no_duplicate_noise(cls, steps: list[ToolCall]) -> list[ToolCall]:
        seen: set[str] = set()
        unique: list[ToolCall] = []
        for step in steps:
            key = f"{step.tool}:{sorted(step.args.items())!r}"
            if key not in seen:
                seen.add(key)
                unique.append(step)
        return unique


class Evidence(BaseModel):
    """A tool result, kept alongside the numbers it is allowed to justify."""

    tool: str
    summary: str
    brief: str = ""  # one-line version used when composing the final answer
    values: dict[str, float] = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)
    ok: bool = True


class ClaimCheck(BaseModel):
    claim: str
    supported: bool
    source: str | None = None


class Verification(BaseModel):
    """The verification agent's verdict on one draft answer."""

    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    ungrounded_numbers: list[float] = Field(default_factory=list)
    claims: list[ClaimCheck] = Field(default_factory=list)
    verifier: str = ""

    def feedback(self) -> str:
        parts = []
        if self.ungrounded_numbers:
            parts.append(f"numbers not found in any tool output: {self.ungrounded_numbers}")
        if self.unsupported_claims:
            parts.append("statements not supported by the evidence: " + " | ".join(self.unsupported_claims))
        return "; ".join(parts)


class AgentAnswer(BaseModel):
    """The final, checked output."""

    answer: str
    evidence: list[Evidence] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
    flags: list[str] = Field(default_factory=list)
    verification: Verification | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    site_id: SiteId = "NL-UTR-01"


class AskResponse(BaseModel):
    run_id: str
    answer: str
    citations: list[str]
    confidence: str
    flags: list[str]
    trace: dict[str, Any]


# --- email drafting (n8n integration) -----------------------------------------

class EmailDraftRequest(BaseModel):
    subject: str = Field(default="", max_length=300)
    body: str = Field(min_length=1, max_length=20000)
    message_id: str | None = Field(default=None, max_length=500)
    site_id: SiteId = "NL-UTR-01"


class DraftDecision(BaseModel):
    approved: bool
    reviewer: str = Field(min_length=1, max_length=120)
    # The one-time secret issued with the draft. Only someone holding it (the
    # reviewer's links, or n8n recording an expiry) can decide.
    approval_token: str = Field(min_length=16, max_length=128)
    edited_reply: str | None = Field(default=None, max_length=20000)
    note: str | None = Field(default=None, max_length=1000)


class DraftRecord(BaseModel):
    draft_id: str
    status: Literal["pending_approval", "approved", "rejected"]
    created_at: float
    message_id: str | None
    subject: str
    question: str  # redacted
    draft_reply: str
    final_reply: str | None = None
    run_id: str
    confidence: str
    flags: list[str]
    citations: list[str]
    requires_attention: bool
    reviewer: str | None = None
    decided_at: float | None = None
    note: str | None = None


class CreatedDraft(DraftRecord):
    """Returned once, when the draft is created. Later reads never include the token."""

    approval_token: str
