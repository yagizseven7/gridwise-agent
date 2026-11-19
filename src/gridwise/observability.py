"""Tracing and metrics.

An agent that you cannot inspect is an agent you cannot operate. Every run writes
one JSON object per step to a JSONL file: which node ran, which tool was called
with what arguments, the (redacted) prompt and completion of every model call,
how long it took, how many tokens it burned, what it cost and whether it failed.
The last step of every run records the answer, its sources, flags and confidence.

Personal data is redacted from every string before the record is written.

In production this same `emit()` would push to CloudWatch / Azure Monitor /
OpenTelemetry instead of a local file. The call sites would not change.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .config import settings
from .pii import redact_obj


# The API serves requests on a thread pool; one lock keeps concurrent runs from
# interleaving partial lines in the shared JSONL file.
_WRITE_LOCK = threading.Lock()


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class StepRecord:
    run_id: str
    step: int
    kind: str  # "guardrail" | "router" | "tool" | "synthesis" | "verification" | "answer"
    name: str
    latency_ms: float
    ok: bool
    input_tokens: int = 0
    output_tokens: int = 0
    cost_eur: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ts": time.time(),
            "run_id": self.run_id,
            "step": self.step,
            "kind": self.kind,
            "name": self.name,
            "latency_ms": round(self.latency_ms, 2),
            "ok": self.ok,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_eur": round(self.cost_eur, 6),
            "detail": self.detail,
        }


def token_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1_000_000 * settings.price_in_per_mtok
        + output_tokens / 1_000_000 * settings.price_out_per_mtok
    )


class Tracer:
    """Collects step records for a single run and appends them to the trace log."""

    def __init__(self, run_id: str | None = None, trace_path: Path | None = None):
        self.run_id = run_id or new_run_id()
        self.trace_path = trace_path or settings.trace_path
        self.records: list[StepRecord] = []
        self._step = 0

    @contextmanager
    def step(self, kind: str, name: str, **detail: Any) -> Iterator[dict[str, Any]]:
        """Times a block of work and records it, even when it raises."""
        self._step += 1
        started = time.perf_counter()
        # The caller can drop extra fields (tokens, row counts) into this dict.
        sink: dict[str, Any] = {"detail": dict(detail)}
        ok = True
        try:
            yield sink
        except Exception as exc:  # noqa: BLE001 - we re-raise after recording
            ok = False
            sink["detail"]["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            in_tok = int(sink.get("input_tokens", 0))
            out_tok = int(sink.get("output_tokens", 0))
            detail = sink["detail"]
            if settings.log_prompts:
                for key in ("prompt", "completion"):
                    if key in sink:
                        detail[key] = str(sink[key])[: settings.log_max_chars]
            detail = redact_obj(detail)
            record = StepRecord(
                run_id=self.run_id,
                step=self._step,
                kind=kind,
                name=name,
                latency_ms=(time.perf_counter() - started) * 1000,
                # A step can also report a non-exception failure (a blocked input,
                # an unverified draft, a handled tool error) by setting sink["ok"].
                ok=ok and bool(sink.get("ok", True)),
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_eur=token_cost(in_tok, out_tok),
                detail=detail,
            )
            self.records.append(record)
            self._write(record)

    @staticmethod
    def record_llm(sink: dict[str, Any], response: Any) -> None:
        """Copy usage, provider, prompt and completion of one model call into a step."""
        sink["input_tokens"] = response.input_tokens
        sink["output_tokens"] = response.output_tokens
        sink["prompt"] = response.prompt
        sink["completion"] = response.text
        sink["detail"]["provider"] = response.provider

    def _write(self, record: StepRecord) -> None:
        try:
            line = json.dumps(record.as_dict(), default=str) + "\n"
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with _WRITE_LOCK, self.trace_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError:
            # Telemetry must never take the request down.
            pass

    # --- run level roll-ups -------------------------------------------------
    @property
    def total_cost_eur(self) -> float:
        return round(sum(r.cost_eur for r in self.records), 6)

    @property
    def total_latency_ms(self) -> float:
        return round(sum(r.latency_ms for r in self.records), 2)

    @property
    def total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self.records)

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "steps": len(self.records),
            "tools_used": [r.name for r in self.records if r.kind == "tool"],
            "total_latency_ms": self.total_latency_ms,
            "total_tokens": self.total_tokens,
            "total_cost_eur": self.total_cost_eur,
            "failed_steps": [r.name for r in self.records if not r.ok],
            "nodes": [r.name for r in self.records],
        }


def read_traces(trace_path: Path | None = None) -> list[dict[str, Any]]:
    path = trace_path or settings.trace_path
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def aggregate_metrics(trace_path: Path | None = None) -> dict[str, Any]:
    """Service-level metrics, the kind you would put on a dashboard."""
    rows = read_traces(trace_path)
    if not rows:
        return {"runs": 0, "steps": 0}

    runs: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        runs.setdefault(row["run_id"], []).append(row)

    run_latencies = sorted(sum(r["latency_ms"] for r in steps) for steps in runs.values())
    tool_rows = [r for r in rows if r["kind"] == "tool"]
    tool_counts: dict[str, int] = {}
    for r in tool_rows:
        tool_counts[r["name"]] = tool_counts.get(r["name"], 0) + 1

    answer_rows = [r for r in rows if r["kind"] == "answer"]
    flag_counts: dict[str, int] = {}
    for r in answer_rows:
        for flag in r["detail"].get("flags", []):
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    first_checks = [r for r in rows if r["kind"] == "verification" and r["name"].endswith("_1")]

    def pct(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        idx = min(len(values) - 1, int(round(q * (len(values) - 1))))
        return round(values[idx], 2)

    return {
        "runs": len(runs),
        "steps": len(rows),
        "step_failure_rate": round(sum(1 for r in rows if not r["ok"]) / len(rows), 4),
        "tool_failure_rate": round(sum(1 for r in tool_rows if not r["ok"]) / len(tool_rows), 4) if tool_rows else 0.0,
        "latency_ms_p50": pct(run_latencies, 0.50),
        "latency_ms_p95": pct(run_latencies, 0.95),
        "total_cost_eur": round(sum(r["cost_eur"] for r in rows), 4),
        "avg_cost_per_run_eur": round(sum(r["cost_eur"] for r in rows) / len(runs), 6),
        "tool_calls": tool_counts,
        "flag_counts": flag_counts,
        "blocked_runs": sum(1 for r in answer_rows if "prompt_injection_blocked" in r["detail"].get("flags", [])),
        "first_draft_grounded_rate": round(
            sum(1 for r in first_checks if r["ok"]) / len(first_checks), 4) if first_checks else None,
    }
