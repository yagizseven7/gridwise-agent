"""Regression evals: a golden set of questions with the behaviour each must show.

Unit tests check functions. This checks *answers*: does the router pick the right
tools, does the answer cite the right sections and contain the right facts, does
it leave out what it must leave out, and does the verification agent accept the
first draft. Run it after every change to a prompt, a tool, the retriever or the
routing rules; a drop in the pass rate is a regression.

Each line of `evals/golden.jsonl` is one case. Supported expectations:

    expect_tools          every one of these tools must run
    forbid_tools          none of these may run
    expect_order          these tools must run in this order
    must_cite             each string must appear in at least one citation
    must_contain          each string must appear in the answer (case-insensitive)
    must_not_contain      none of these may appear in the answer
    expect_flags          each flag must be raised
    forbid_flags          none of these flags may be raised
    forbid_flag_prefixes  no flag may start with these
    expect_blocked        true: the input guard must refuse; false: it must not
    expect_confidence     the confidence must be one of these
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import settings
from .orchestrator import Agent


@dataclass
class CaseResult:
    id: str
    category: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    confidence: str = ""
    first_draft_grounded: bool | None = None
    citation_recall: float | None = None
    latency_ms: float = 0.0
    tokens: int = 0


def load_cases(path: Path | None = None) -> list[dict[str, Any]]:
    lines = Path(path or settings.evals_path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip() and not line.lstrip().startswith("//")]


def score_case(case: dict[str, Any], agent: Agent) -> CaseResult:
    answer, tracer = agent.run(case["question"], site_id=case.get("site_id", "NL-UTR-01"))
    tools = [r.name for r in tracer.records if r.kind == "tool"]
    text = answer.answer.lower()
    failures: list[str] = []

    for tool in case.get("expect_tools", []):
        if tool not in tools:
            failures.append(f"expected tool {tool} (ran {tools})")
    for tool in case.get("forbid_tools", []):
        if tool in tools:
            failures.append(f"forbidden tool {tool} ran")
    order = case.get("expect_order", [])
    if order and all(t in tools for t in order):
        if [tools.index(t) for t in order] != sorted(tools.index(t) for t in order):
            failures.append(f"tools out of order: {tools}")

    cite_hits = [c for c in case.get("must_cite", []) if any(c in got for got in answer.citations)]
    for c in case.get("must_cite", []):
        if c not in cite_hits:
            failures.append(f"missing citation {c!r}")
    for s in case.get("must_contain", []):
        if s.lower() not in text:
            failures.append(f"answer lacks {s!r}")
    for s in case.get("must_not_contain", []):
        if s.lower() in text:
            failures.append(f"answer contains forbidden {s!r}")

    for f in case.get("expect_flags", []):
        if f not in answer.flags:
            failures.append(f"expected flag {f}")
    for f in case.get("forbid_flags", []):
        if f in answer.flags:
            failures.append(f"forbidden flag {f}")
    for prefix in case.get("forbid_flag_prefixes", []):
        if any(f.startswith(prefix) for f in answer.flags):
            failures.append(f"forbidden flag prefix {prefix}")

    blocked = "prompt_injection_blocked" in answer.flags
    if "expect_blocked" in case and blocked != case["expect_blocked"]:
        failures.append(f"expected blocked={case['expect_blocked']}, got {blocked}")
    if "expect_confidence" in case and answer.confidence not in case["expect_confidence"]:
        failures.append(f"confidence {answer.confidence} not in {case['expect_confidence']}")

    first_check = next((r for r in tracer.records if r.kind == "verification"), None)
    summary = tracer.summary()
    return CaseResult(
        id=case["id"],
        category=case.get("category", ""),
        passed=not failures,
        failures=failures,
        tools=tools,
        citations=answer.citations,
        flags=answer.flags,
        confidence=answer.confidence,
        first_draft_grounded=first_check.ok if first_check else None,
        citation_recall=(len(cite_hits) / len(case["must_cite"])) if case.get("must_cite") else None,
        latency_ms=summary["total_latency_ms"],
        tokens=summary["total_tokens"],
    )


def run_suite(path: Path | None = None, agent: Agent | None = None) -> dict[str, Any]:
    agent = agent or Agent()
    results = [score_case(case, agent) for case in load_cases(path)]

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    routed = [r for r in results if not any(f.startswith(("expected tool", "forbidden tool", "tools out")) for f in r.failures)]
    by_category: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = by_category.setdefault(r.category, {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(r.passed)

    return {
        "model": agent.llm.name,
        "cases": len(results),
        "passed": sum(r.passed for r in results),
        "pass_rate": mean([float(r.passed) for r in results]),
        "routing_accuracy": round(len(routed) / len(results), 4) if results else None,
        "citation_recall": mean([r.citation_recall for r in results if r.citation_recall is not None]),
        "first_draft_grounded_rate": mean([float(r.first_draft_grounded) for r in results if r.first_draft_grounded is not None]),
        "avg_tokens_per_case": mean([float(r.tokens) for r in results]),
        "by_category": by_category,
        "results": [asdict(r) for r in results],
    }
