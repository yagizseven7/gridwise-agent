"""The production surface.

The same LangGraph pipeline is served here behind a typed HTTP API:

    POST /ask                    ask a question, get a checked answer + trace summary
    POST /email/draft            n8n: turn an incoming email into a draft reply
    GET  /drafts/{id}            n8n / reviewer: read a draft
    POST /drafts/{id}/decision   n8n: record the human approve / reject decision
    GET  /health                 liveness + which model is active
    GET  /metrics                latency percentiles, failure rate, cost, guardrail counts
    GET  /traces/{run_id}        every step of one run
    GET  /graph                  the pipeline as a Mermaid diagram

Run locally:  uvicorn gridwise.api:app --reload
Docs:         http://localhost:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException

from . import __version__
from .config import settings
from .data import ensure_database
from .drafts import DraftConflict, DraftForbidden, DraftStore
from .observability import aggregate_metrics, read_traces
from .orchestrator import Agent
from .retrieval import embedder_name
from .schemas import AskRequest, AskResponse, CreatedDraft, DraftDecision, DraftRecord, EmailDraftRequest

_agent: Agent | None = None
_drafts: DraftStore | None = None


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent()
    return _agent


def get_drafts() -> DraftStore:
    global _drafts
    if _drafts is None:
        _drafts = DraftStore()
    return _drafts


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Fail fast at boot rather than on the first user request.
    ensure_database()
    get_agent()
    get_drafts()
    yield


app = FastAPI(
    title="GridWise agentic energy analyst",
    version=__version__,
    description="Answers energy questions from telemetry, a forecaster and client documents, "
                "with guardrails, a verification agent and human-approved email replies.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "pipeline": "langgraph",
        "provider": settings.resolved_provider,
        "model": settings.model_label,
        "embeddings": embedder_name(),
        "mode": "offline_stub" if settings.offline else "live_model",
        "max_steps": settings.max_steps,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        answer, tracer = get_agent().run(request.question, site_id=request.site_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"agent_failure: {type(exc).__name__}") from exc

    return AskResponse(
        run_id=tracer.run_id,
        answer=answer.answer,
        citations=answer.citations,
        confidence=answer.confidence,
        flags=answer.flags,
        trace=tracer.summary(),
    )


@app.post("/email/draft", response_model=CreatedDraft)
def email_draft(request: EmailDraftRequest) -> CreatedDraft:
    try:
        return get_drafts().create(request, get_agent())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"draft_failure: {type(exc).__name__}") from exc


@app.get("/drafts/{draft_id}", response_model=DraftRecord)
def get_draft(draft_id: str) -> DraftRecord:
    record = get_drafts().get(draft_id)
    if record is None:
        raise HTTPException(status_code=404, detail="draft not found")
    return record


@app.post("/drafts/{draft_id}/decision", response_model=DraftRecord)
def decide_draft(draft_id: str, decision: DraftDecision) -> DraftRecord:
    try:
        return get_drafts().decide(draft_id, decision)
    except KeyError:
        raise HTTPException(status_code=404, detail="draft not found") from None
    except DraftForbidden:
        raise HTTPException(status_code=403, detail="invalid approval token") from None
    except DraftConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    """Service-level roll-up: latency percentiles, failure rate, cost, guardrail counts."""
    return aggregate_metrics()


@app.get("/traces/{run_id}")
def trace(run_id: str) -> dict[str, Any]:
    rows = [row for row in read_traces() if row["run_id"] == run_id]
    if not rows:
        raise HTTPException(status_code=404, detail="run_id not found")
    return {"run_id": run_id, "steps": rows}


@app.get("/graph")
def graph() -> dict[str, str]:
    return {"mermaid": get_agent().graph.mermaid()}
