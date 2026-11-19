"""The regression gate: the golden set must pass in full against the deterministic stand-in."""

from __future__ import annotations

from gridwise.evals import run_suite


def test_golden_set_passes() -> None:
    report = run_suite()
    failures = {r["id"]: r["failures"] for r in report["results"] if not r["passed"]}
    assert failures == {}
    assert report["routing_accuracy"] == 1.0
    assert report["citation_recall"] == 1.0
