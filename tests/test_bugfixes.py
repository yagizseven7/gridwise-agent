"""Regression tests for bugs found in review."""

from __future__ import annotations

import json
import sqlite3
import threading

from gridwise import drafts as drafts_module
from gridwise import guardrails
from gridwise.drafts import DraftStore
from gridwise.llm import RuleBasedLLM
from gridwise.observability import Tracer
from gridwise.orchestrator import Agent
from gridwise.schemas import EmailDraftRequest


def test_sqlite_connections_are_closed(tmp_path, monkeypatch) -> None:
    opened, closed = [], []
    real_connect = sqlite3.connect

    class Tracked(sqlite3.Connection):
        def close(self) -> None:
            closed.append(self)
            super().close()

    def connect(*args, **kwargs):
        con = real_connect(*args, factory=Tracked, **kwargs)
        opened.append(con)
        return con

    monkeypatch.setattr(drafts_module.sqlite3, "connect", connect)
    store = DraftStore(tmp_path / "d.sqlite")
    created = store.create(EmailDraftRequest(subject="cold spell", body="What is the cold spell protocol?"),
                           Agent(llm=RuleBasedLLM()))
    store.get(created.draft_id)
    assert opened and len(closed) == len(opened)


def test_concurrent_runs_never_corrupt_the_trace_file(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"

    def work(i: int) -> None:
        tracer = Tracer(trace_path=path)
        for j in range(40):
            with tracer.step("tool", f"t{i}", payload="x" * 2000, j=j):
                pass

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = path.read_text().splitlines()
    assert len(lines) == 8 * 40
    assert all(json.loads(line)["kind"] == "tool" for line in lines)


def test_numbers_in_the_question_are_not_evidence() -> None:
    evidence = [{"tool": "forecast_demand", "summary": "peak 426.1 kW", "values": {"peak_kw": 426.1}}]
    assert guardrails.check_numeric_grounding("Yes, the peak is 812 kW.", evidence, question="Is the peak 812 kW?") == [812.0]


def test_time_windows_from_the_question_may_be_echoed() -> None:
    evidence = [{"tool": "query_sensor_data", "summary": "peak kw 461.5", "values": {"peak_kw": 461.5}}]
    answer = "Over the last 30 days the highest peak was 461.5 kW."
    assert guardrails.check_numeric_grounding(answer, evidence, question="highest peak in the last 30 days?") == []
