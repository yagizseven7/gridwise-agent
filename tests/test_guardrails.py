from __future__ import annotations

from gridwise import guardrails

EVIDENCE = [
    {
        "tool": "forecast_demand",
        "summary": "Forecast peak 512.4 kW, mean 210.0 kW, hold-out MAPE 8.2%.",
        "values": {"peak_kw": 512.4, "mean_kw": 210.0, "mape": 0.082},
        "citations": ["model:demand_forecaster_v0.1"],
    }
]


class TestToolCallValidation:
    def test_accepts_a_well_formed_call(self) -> None:
        call, error = guardrails.validate_tool_call(
            {"tool": "forecast_demand", "args": {"site_id": "NL-UTR-01"}}
        )
        assert error is None
        assert call is not None and call.tool == "forecast_demand"

    def test_rejects_an_unknown_tool(self) -> None:
        call, error = guardrails.validate_tool_call({"tool": "delete_everything", "args": {}})
        assert call is None
        assert error is not None


class TestNumericGrounding:
    def test_grounded_numbers_pass(self) -> None:
        answer = "The forecast peak is 512.4 kW, with a mean of 210.0 kW and MAPE of 8.2%."
        assert guardrails.check_numeric_grounding(answer, EVIDENCE) == []

    def test_invented_numbers_are_caught(self) -> None:
        answer = "The peak will hit 812 kW and cost you EUR 5,376."
        caught = guardrails.check_numeric_grounding(answer, EVIDENCE)
        assert 812.0 in caught
        assert 5376.0 in caught

    def test_clock_times_are_not_treated_as_quantities(self) -> None:
        answer = "Peak of 512.4 kW expected at 17:30 on 2026-09-01."
        assert guardrails.check_numeric_grounding(answer, EVIDENCE) == []

    def test_rounding_within_tolerance_is_accepted(self) -> None:
        assert guardrails.check_numeric_grounding("About 512 kW.", EVIDENCE) == []


class TestQualityGates:
    def test_degraded_forecast_is_flagged(self) -> None:
        evidence = [{"tool": "forecast_demand", "summary": "", "values": {"mape": 0.19}}]
        assert "forecast_quality_below_threshold" in guardrails.quality_flags(evidence)

    def test_peak_near_contract_triggers_review(self) -> None:
        evidence = [{"tool": "estimate_imbalance_cost", "summary": "", "values": {"capacity_used_pct": 95.0}}]
        assert "peak_shaving_trigger_reached" in guardrails.quality_flags(evidence)

    def test_healthy_run_has_no_flags(self) -> None:
        assert guardrails.quality_flags(EVIDENCE) == []
