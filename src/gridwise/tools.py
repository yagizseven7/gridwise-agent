"""The tools.

The router agent cannot touch the database, the forecaster or the documents
directly -- it can only ask for one of the functions below, by name, with
arguments that are validated first.

Every tool returns the same envelope:

    {
      "summary":   one paragraph a language model can read,
      "brief":     one line, used when composing the final answer,
      "values":    {name: number} -- the only numbers allowed in the final answer,
      "citations": where this came from,
      "raw":       full detail, kept for the trace and the API response
    }

Adding a capability to the agent means adding a function here and its name to
`schemas.ToolName`. No new control flow.
"""

from __future__ import annotations

from typing import Any, Callable

from . import data, forecasting, retrieval, summarize
from .config import settings
from .guardrails import detect_prompt_injection

# Tariff constants, taken from knowledge/tariffs_and_imbalance.md
CAPACITY_SURCHARGE_EUR_PER_KW = 28.0
IMBALANCE_EUR_PER_MWH = 90.0

SCHEMA_HINT = (
    "Table readings(ts TIMESTAMPTZ, site_id VARCHAR, demand_kw DOUBLE, "
    "pv_kw DOUBLE, temp_c DOUBLE, is_weekend INTEGER). "
    "Sites: NL-UTR-01, NL-AMS-02. Half-hourly resolution."
)


def _screen(texts_with_ids: list[tuple[str, str]]) -> tuple[set[str], list[dict[str, Any]]]:
    """Indirect prompt-injection check on retrieved text. Returns ids to drop."""
    dropped: set[str] = set()
    report: list[dict[str, Any]] = []
    for ident, text in texts_with_ids:
        verdict = detect_prompt_injection(text)
        if verdict.blocked:
            dropped.add(ident)
            report.append({"citation": ident, "score": verdict.score, "rules": verdict.rules})
    return dropped, report


def query_sensor_data(sql: str, **_: Any) -> dict[str, Any]:
    """Run one read-only SQL query against the telemetry table."""
    frame = data.run_sql(sql)
    rows = frame.head(20).to_dict(orient="records")

    values: dict[str, float] = {}
    if len(frame) == 1:
        for column, value in frame.iloc[0].items():
            try:
                values[str(column)] = round(float(value), 3)
            except (TypeError, ValueError):
                continue

    if values:
        rendered = ", ".join(f"{k.replace('_', ' ')} {v:g}" for k, v in values.items())
        summary = f"Measured from telemetry: {rendered}."
    else:
        preview = rows[0] if rows else {}
        summary = (
            f"Query returned {len(frame)} rows over columns {list(frame.columns)}; "
            f"first row: {preview}."
        )

    return {
        "summary": summary,
        "brief": summary,
        "values": values,
        "citations": ["duckdb:readings"],
        "raw": {"sql": sql, "row_count": int(len(frame)), "rows": rows},
    }


def forecast_demand(site_id: str = "NL-UTR-01", horizon_hours: int = 24, **_: Any) -> dict[str, Any]:
    """Train and backtest the demand forecaster for one site."""
    data.validate_site(site_id)
    horizon_hours = int(max(1, min(48, int(horizon_hours))))
    series = data.load_site_series(site_id)
    result = forecasting.forecast(series, site_id=site_id, horizon_hours=horizon_hours)

    degraded = result.mape > settings.mape_alert_threshold
    summary = (
        f"{site_id}: forecast peak {result.peak_kw} kW at {result.peak_at} UTC across the "
        f"{result.horizon_hours} h horizon, mean {result.mean_kw} kW, {result.total_kwh} kWh in total. "
        f"Walk-forward hold-out MAPE {result.mape:.1%} against a seasonal-naive baseline of "
        f"{result.baseline_mape:.1%} ({result.improvement_pct}% better), trained on "
        f"{result.n_train_rows} rows."
    )
    brief = (
        f"Forecast peak {result.peak_kw} kW at {result.peak_at} UTC "
        f"({result.horizon_hours} h horizon), mean {result.mean_kw} kW, "
        f"{result.total_kwh} kWh total, hold-out MAPE {result.mape:.1%} vs baseline "
        f"{result.baseline_mape:.1%}."
    )
    if degraded:
        note = " Forecast quality is below the acceptance threshold; flagged for human review."
        summary += note
        brief += note

    return {
        "summary": summary,
        "brief": brief,
        "values": {
            "peak_kw": result.peak_kw,
            "mean_kw": result.mean_kw,
            "total_kwh": result.total_kwh,
            "mape": result.mape,
            "baseline_mape": result.baseline_mape,
            "improvement_pct": result.improvement_pct,
            "horizon_hours": float(result.horizon_hours),
        },
        "citations": ["model:demand_forecaster_v0.1"],
        "raw": result.to_dict() | {"degraded": degraded},
    }


def search_playbook(query: str, k: int = 3, **_: Any) -> dict[str, Any]:
    """Retrieve the client rules that apply to this question."""
    passages = retrieval.search(query, k=int(max(1, min(5, int(k)))))
    dropped, quarantined = _screen([(p.citation, p.text) for p in passages])
    passages = [p for p in passages if p.citation not in dropped]

    if not passages:
        return {
            "summary": "No matching passage found in the client knowledge base.",
            "brief": "No client rule found for this question.",
            "values": {},
            "citations": [],
            "raw": {"query": query, "passages": [], "quarantined": quarantined},
        }

    joined = " | ".join(f"[{p.citation}] {p.text}" for p in passages)
    top = passages[0]
    return {
        "summary": f"Retrieved {len(passages)} passage(s): {joined}",
        "brief": f"Client rule [{top.citation}]: {top.text}",
        "values": {},
        "citations": [p.citation for p in passages],
        "raw": {
            "query": query,
            "passages": [{"citation": p.citation, "score": p.score, "text": p.text} for p in passages],
            "quarantined": quarantined,
        },
    }


def summarize_documents(topic: str = "", source: str | None = None, max_sentences: int = 4, **_: Any) -> dict[str, Any]:
    """Extractive summary of one client document, or of the passages matching a topic."""
    sentences = summarize.summarize(topic=topic, source=source, max_sentences=int(max(1, min(8, int(max_sentences)))))
    dropped, quarantined = _screen([(f"{s.citation}@{s.order}", s.text) for s in sentences])
    sentences = [s for s in sentences if f"{s.citation}@{s.order}" not in dropped]

    label = summarize.resolve_source(source) or (f"'{topic}'" if topic else "the knowledge base")
    if not sentences:
        return {
            "summary": f"Nothing to summarise for {label}.",
            "brief": f"No client document matched {label}.",
            "values": {},
            "citations": [],
            "raw": {"topic": topic, "source": source, "sentences": [], "quarantined": quarantined},
        }

    body = " ".join(s.text for s in sentences)
    citations = list(dict.fromkeys(s.citation for s in sentences))
    return {
        "summary": f"Summary of {label}: " + " ".join(f"[{s.citation}] {s.text}" for s in sentences),
        "brief": f"Summary of {label}: {body}",
        "values": {},
        "citations": citations,
        "raw": {
            "topic": topic,
            "source": source,
            "sentences": [{"text": s.text, "citation": s.citation} for s in sentences],
            "quarantined": quarantined,
        },
    }


def estimate_imbalance_cost(
    peak_kw: float | None = None,
    contracted_kw: float | None = None,
    deviation_mwh: float = 0.0,
    **_: Any,
) -> dict[str, Any]:
    """Deterministic money calculation. Never ask a language model to do arithmetic."""
    if peak_kw is None or contracted_kw is None:
        raise ValueError(
            "estimate_imbalance_cost needs peak_kw and contracted_kw; run forecast_demand "
            "or query_sensor_data (selecting a column named peak_kw) first."
        )
    peak_kw = float(peak_kw)
    contracted_kw = float(contracted_kw)
    over_kw = max(0.0, peak_kw - contracted_kw)
    capacity_surcharge = round(over_kw * CAPACITY_SURCHARGE_EUR_PER_KW, 2)
    imbalance_cost = round(float(deviation_mwh) * IMBALANCE_EUR_PER_MWH, 2)
    total = round(capacity_surcharge + imbalance_cost, 2)
    headroom_pct = round(peak_kw / contracted_kw * 100, 1) if contracted_kw else 0.0

    verdict = (
        "no capacity surcharge is due"
        if over_kw == 0
        else f"the peak exceeds contract by {over_kw:.1f} kW"
    )
    line = (
        f"A peak of {peak_kw:g} kW uses {headroom_pct}% of the {contracted_kw:g} kW contracted capacity, "
        f"so {verdict}; capacity surcharge EUR {capacity_surcharge:,.2f}, imbalance exposure "
        f"EUR {imbalance_cost:,.2f}, total exposure EUR {total:,.2f}."
    )
    return {
        "summary": line,
        "brief": line,
        "values": {
            "peak_kw": peak_kw,
            "contracted_kw": contracted_kw,
            "over_kw": round(over_kw, 1),
            "capacity_surcharge_eur": capacity_surcharge,
            "imbalance_cost_eur": imbalance_cost,
            "total_cost_eur": total,
            "capacity_used_pct": headroom_pct,
        },
        "citations": ["tariffs_and_imbalance.md#Contracted capacity and peak charges"],
        "raw": {
            "peak_kw": peak_kw,
            "contracted_kw": contracted_kw,
            "rate_eur_per_kw": CAPACITY_SURCHARGE_EUR_PER_KW,
            "rate_eur_per_mwh": IMBALANCE_EUR_PER_MWH,
        },
    }


# --- registry ---------------------------------------------------------------
# The description strings are what the router sees. They are part of the prompt,
# so they are written for a reader, not for a compiler.
TOOLS: dict[str, dict[str, Any]] = {
    "query_sensor_data": {
        "fn": query_sensor_data,
        "description": (
            "SQL lookup: run one read-only SELECT against half-hourly building telemetry. "
            f"{SCHEMA_HINT} Use it for historical facts: averages, peaks, counts, comparisons. "
            "Name a peak column peak_kw if a cost calculation follows."
        ),
        "args": {"sql": "string, a single SELECT statement"},
        "example": {"sql": "SELECT max(demand_kw) AS peak_kw FROM readings WHERE site_id = 'NL-UTR-01'"},
    },
    "forecast_demand": {
        "fn": forecast_demand,
        "description": (
            "Train and backtest the demand forecaster and return the predicted peak, mean, "
            "total energy and the hold-out error against a seasonal-naive baseline. "
            "Use it for any question about the future."
        ),
        "args": {"site_id": "string", "horizon_hours": "integer, 1-48"},
        "example": {"site_id": "NL-UTR-01", "horizon_hours": 24},
    },
    "search_playbook": {
        "fn": search_playbook,
        "description": (
            "Document search: retrieve passages from the client knowledge base (tariff handbook, "
            "site operations playbook, model card) with citations. Use it for rules, thresholds, "
            "procedures, responsibilities and site specifications. For a compound question, "
            "call it once per sub-question."
        ),
        "args": {"query": "string", "k": "integer, 1-5"},
        "example": {"query": "peak shaving trigger threshold", "k": 3},
    },
    "summarize_documents": {
        "fn": summarize_documents,
        "description": (
            "Summarisation: an extractive summary (verbatim, cited sentences) of one client "
            "document or of the passages about a topic. Use it when the user asks for a summary, "
            "overview or the key points."
        ),
        "args": {"topic": "string, optional", "source": "document name, optional", "max_sentences": "integer, 1-8"},
        "example": {"source": "tariffs_and_imbalance.md", "max_sentences": 4},
    },
    "estimate_imbalance_cost": {
        "fn": estimate_imbalance_cost,
        "description": (
            "Compute the euro impact of a peak: capacity surcharge above contracted "
            "capacity plus imbalance exposure. Use it whenever the question involves cost. "
            "Pass peak_kw as null to take it from an earlier step."
        ),
        "args": {
            "peak_kw": "number or null",
            "contracted_kw": "number",
            "deviation_mwh": "number, optional",
        },
        "example": {"peak_kw": None, "contracted_kw": 620, "deviation_mwh": 1.5},
    },
}


def tool_catalogue() -> str:
    """The tool list, rendered for the router prompt."""
    lines = []
    for name, spec in TOOLS.items():
        lines.append(f"- {name}({', '.join(spec['args'])}): {spec['description']}")
        lines.append(f"  example args: {spec['example']}")
    return "\n".join(lines)


def get_tool(name: str) -> Callable[..., dict[str, Any]]:
    if name not in TOOLS:
        raise KeyError(f"Unknown tool '{name}'. Available: {sorted(TOOLS)}")
    return TOOLS[name]["fn"]
