"""Run the whole workflow from the command line.

    python scripts/run_demo.py
    python scripts/run_demo.py "What will the peak be tomorrow and what will it cost?"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridwise.config import settings  # noqa: E402
from gridwise.data import build_database  # noqa: E402
from gridwise.observability import aggregate_metrics  # noqa: E402
from gridwise.orchestrator import ask  # noqa: E402

DEFAULT_QUESTIONS = [
    "What was the average and peak demand at this site over the whole history?",
    "What peak demand should we expect in the next 24 hours?",
    "Will tomorrow's forecast peak cost us anything against our contracted capacity?",
    "When are we supposed to start peak shaving, and who signs it off?",
    "Summarize the tariff handbook.",
]

RULE = "=" * 78


def main() -> None:
    questions = sys.argv[1:] or DEFAULT_QUESTIONS

    if not settings.db_path.exists():
        print("Building the telemetry database (first run only)...")
        build_database()

    mode = "offline deterministic stub" if settings.offline else f"live model {settings.model_label}"
    print(f"\nGridWise agent — mode: {mode}\n")

    for question in questions:
        print(RULE)
        print(f"Q: {question}\n")
        answer, tracer = ask(question)

        print(f"A: {answer.answer}\n")
        print(f"   confidence : {answer.confidence}")
        print(f"   citations  : {', '.join(answer.citations) or 'none'}")
        print(f"   flags      : {', '.join(answer.flags) or 'none'}")
        summary = tracer.summary()
        print(f"   tools used : {' -> '.join(summary['tools_used']) or 'none'}")
        print(f"   graph path : {' -> '.join(summary['nodes'])}")
        print(
            f"   trace      : run {summary['run_id']}, {summary['steps']} steps, "
            f"{summary['total_latency_ms']} ms, {summary['total_tokens']} tokens, "
            f"EUR {summary['total_cost_eur']}"
        )
        print()

    print(RULE)
    print("Service metrics across all runs in the trace log:")
    for key, value in aggregate_metrics().items():
        print(f"   {key}: {value}")
    print(f"\nFull traces: {settings.trace_path}")


if __name__ == "__main__":
    main()
