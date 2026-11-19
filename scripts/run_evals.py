"""Run the regression eval set and print a scorecard.

    python scripts/run_evals.py                    # all cases, fail if any case fails
    python scripts/run_evals.py --min-pass-rate 0.9   # e.g. against a live, sampled model
    python scripts/run_evals.py --report data/eval_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridwise.evals import run_suite  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--min-pass-rate", type=float, default=1.0)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    report = run_suite(args.cases)
    print(f"model: {report['model']}\n")
    print(f"{'case':<10} {'result':<6} {'tools':<55} detail")
    print("-" * 110)
    for r in report["results"]:
        status = "PASS" if r["passed"] else "FAIL"
        tools = " -> ".join(r["tools"]) or "(refused)"
        print(f"{r['id']:<10} {status:<6} {tools[:55]:<55} {'; '.join(r['failures'])[:80]}")
    print("-" * 110)
    for key in ("passed", "cases", "pass_rate", "routing_accuracy", "citation_recall",
                "first_draft_grounded_rate", "avg_tokens_per_case"):
        print(f"{key:<26} {report[key]}")
    print(f"{'by_category':<26} {json.dumps(report['by_category'])}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport written to {args.report}")
    return 0 if (report["pass_rate"] or 0) >= args.min_pass_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
