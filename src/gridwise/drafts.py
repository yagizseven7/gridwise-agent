"""Email drafts awaiting human approval.

The n8n workflow in `integrations/n8n/` turns an incoming email into a call to
`POST /email/draft`. The agent answers the question, and the answer is stored
here as a *draft* with status `pending_approval`. Nothing is sent from this
service: n8n only emails the customer after a reviewer approves the draft via
`POST /drafts/{id}/decision`, and only the text returned by that call.

Each draft is issued a one-time approval token (returned once, stored only as a
SHA-256 hash). A decision without the matching token is a 403, so knowing or
guessing a draft id -- or n8n's sequential resume URL -- is not enough to approve.

State machine (a decision is final; deciding twice is a 409):

    pending_approval ──approve──► approved   (final_reply = edited text or draft)
                     └─reject───► rejected

Privacy: the sender's address is not stored. n8n keeps it in the paused
workflow execution so it can reply; this service only sees the redacted body.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import time
import uuid
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Iterator

from .config import settings
from .orchestrator import Agent
from .pii import redact
from .schemas import CreatedDraft, DraftDecision, DraftRecord, EmailDraftRequest

_QUOTED = re.compile(r"^(>|On .+ wrote:|-----Original Message-----|-- ?$|Sent from my )", re.M)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS drafts (
    draft_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    message_id TEXT,
    subject TEXT,
    question TEXT,
    draft_reply TEXT,
    final_reply TEXT,
    run_id TEXT,
    confidence TEXT,
    flags TEXT,
    citations TEXT,
    requires_attention INTEGER,
    reviewer TEXT,
    decided_at REAL,
    note TEXT,
    token_hash TEXT
)
"""


class DraftConflict(Exception):
    """The draft has already been decided."""


class DraftForbidden(Exception):
    """The approval token is missing or wrong."""


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def email_to_question(subject: str, body: str) -> str:
    """Strip quoted history and signatures; keep subject + the new text."""
    match = _QUOTED.search(body)
    fresh = body[: match.start()] if match else body
    fresh = " ".join(fresh.split())
    subject = re.sub(r"^\s*(re|fw|fwd)\s*:\s*", "", subject or "", flags=re.I).strip()
    question = f"{subject}. {fresh}" if subject and subject.lower() not in fresh.lower() else fresh
    return question[:2000].strip() or subject


class DraftStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or settings.drafts_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute(_SCHEMA)
            columns = {row["name"] for row in con.execute("PRAGMA table_info(drafts)")}
            if "token_hash" not in columns:  # database created by v0.2
                con.execute("ALTER TABLE drafts ADD COLUMN token_hash TEXT")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """One short-lived connection per operation: commit on success, always close.

        (`with sqlite3.connect(...)` alone commits but never closes the connection.)
        """
        with closing(sqlite3.connect(self.path, timeout=10)) as con:
            con.row_factory = sqlite3.Row
            with con:
                yield con

    def create(self, request: EmailDraftRequest, agent: Agent) -> CreatedDraft:
        question = email_to_question(request.subject, request.body)
        answer, tracer = agent.run(question, site_id=request.site_id)
        redacted_question, _ = redact(question)
        redacted_subject, _ = redact(request.subject)

        record = DraftRecord(
            draft_id=uuid.uuid4().hex[:12],
            status="pending_approval",
            created_at=time.time(),
            message_id=request.message_id,
            subject=redacted_subject,
            question=redacted_question,
            draft_reply=answer.answer,
            run_id=tracer.run_id,
            confidence=answer.confidence,
            flags=answer.flags,
            citations=answer.citations,
            # Reviewers see these first: anything the agent was unsure about.
            requires_attention=answer.confidence == "low" or bool(answer.flags),
        )
        token = secrets.token_urlsafe(24)
        with self._connect() as con:
            con.execute(
                "INSERT INTO drafts VALUES (:draft_id, :status, :created_at, :message_id, :subject, :question, "
                ":draft_reply, :final_reply, :run_id, :confidence, :flags, :citations, :requires_attention, "
                ":reviewer, :decided_at, :note, :token_hash)",
                record.model_dump() | {
                    "flags": json.dumps(record.flags),
                    "citations": json.dumps(record.citations),
                    "requires_attention": int(record.requires_attention),
                    "token_hash": _hash_token(token),
                },
            )
        return CreatedDraft(**record.model_dump(), approval_token=token)

    def get(self, draft_id: str) -> DraftRecord | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM drafts WHERE draft_id = ?", [draft_id]).fetchone()
        if row is None:
            return None
        data = dict(row)
        data.pop("token_hash", None)
        data["flags"] = json.loads(data["flags"] or "[]")
        data["citations"] = json.loads(data["citations"] or "[]")
        data["requires_attention"] = bool(data["requires_attention"])
        return DraftRecord.model_validate(data)

    def _token_hash(self, draft_id: str) -> str | None:
        with self._connect() as con:
            row = con.execute("SELECT token_hash FROM drafts WHERE draft_id = ?", [draft_id]).fetchone()
        return row["token_hash"] if row else None

    def decide(self, draft_id: str, decision: DraftDecision) -> DraftRecord:
        record = self.get(draft_id)
        if record is None:
            raise KeyError(draft_id)
        expected = self._token_hash(draft_id) or ""
        if not hmac.compare_digest(_hash_token(decision.approval_token), expected):
            raise DraftForbidden(f"invalid approval token for draft {draft_id}")
        if record.status != "pending_approval":
            raise DraftConflict(f"draft {draft_id} is already {record.status}")

        status = "approved" if decision.approved else "rejected"
        final_reply = None
        if decision.approved:
            final_reply = (decision.edited_reply or record.draft_reply).strip()
        with self._connect() as con:
            # The WHERE clause makes the transition atomic under concurrent clicks.
            cursor = con.execute(
                "UPDATE drafts SET status = ?, final_reply = ?, reviewer = ?, decided_at = ?, note = ? "
                "WHERE draft_id = ? AND status = 'pending_approval'",
                [status, final_reply, decision.reviewer, time.time(), decision.note, draft_id],
            )
            if cursor.rowcount == 0:
                raise DraftConflict(f"draft {draft_id} was decided concurrently")
        return self.get(draft_id)  # type: ignore[return-value]
