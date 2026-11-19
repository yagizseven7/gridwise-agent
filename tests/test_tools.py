from __future__ import annotations

import pytest

from gridwise import data, tools


class TestSqlSafety:
    @pytest.mark.parametrize(
        "sql",
        [
            "DROP TABLE readings",
            "SELECT 1; DELETE FROM readings",
            "INSERT INTO readings VALUES (1)",
            "UPDATE readings SET demand_kw = 0",
            "COPY readings TO 'out.csv'",
        ],
    )
    def test_write_statements_are_rejected(self, sql: str) -> None:
        assert data.is_read_only(sql) is False
        with pytest.raises(PermissionError):
            data.run_sql(sql)

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM read_text('/etc/hostname')",
            "SELECT * FROM read_csv('/etc/passwd')",
            "SELECT * FROM '/etc/passwd'",
        ],
    )
    def test_file_system_reads_are_blocked(self, sql: str) -> None:
        """These pass the keyword allow-list; the connection itself must refuse them."""
        with pytest.raises(Exception) as excinfo:
            data.run_sql(sql)
        assert "readings" not in str(excinfo.value).lower() or "permission" in str(excinfo.value).lower()

    def test_select_is_allowed(self) -> None:
        frame = data.run_sql("SELECT count(*) AS n FROM readings")
        assert int(frame.iloc[0]["n"]) > 0


class TestForecastTool:
    def test_returns_a_usable_envelope(self) -> None:
        result = tools.forecast_demand(site_id="NL-UTR-01", horizon_hours=24)
        assert set(result) >= {"summary", "values", "citations", "raw"}
        assert result["values"]["peak_kw"] > 0
        assert 0 <= result["values"]["mape"] < 1

    def test_beats_the_seasonal_naive_baseline(self) -> None:
        """The whole justification for the model. If this fails, ship the baseline."""
        values = tools.forecast_demand(site_id="NL-UTR-01", horizon_hours=24)["values"]
        assert values["mape"] <= values["baseline_mape"]

    def test_horizon_is_clamped(self) -> None:
        result = tools.forecast_demand(site_id="NL-UTR-01", horizon_hours=999)
        assert result["raw"]["horizon_hours"] == 48


class TestRetrievalTool:
    def test_finds_the_peak_shaving_rule(self) -> None:
        result = tools.search_playbook(query="when do we start peak shaving", k=3)
        assert result["citations"]
        assert "92" in result["summary"]

    def test_every_passage_is_attributable(self) -> None:
        result = tools.search_playbook(query="battery storage capacity", k=3)
        for passage in result["raw"]["passages"]:
            assert "#" in passage["citation"]


class TestSummariseTool:
    def test_document_summary_is_cited(self) -> None:
        result = tools.summarize_documents(source="site_operations_playbook.md", max_sentences=3)
        assert len(result["raw"]["sentences"]) == 3
        assert all(c.startswith("site_operations_playbook.md#") for c in result["citations"])


class TestSiteValidation:
    def test_unknown_site_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            tools.forecast_demand(site_id="NL-XXX-99")


class TestCostTool:
    def test_missing_peak_is_a_clear_error(self) -> None:
        with pytest.raises(ValueError, match="peak_kw"):
            tools.estimate_imbalance_cost(contracted_kw=620)

    def test_no_surcharge_below_contract(self) -> None:
        values = tools.estimate_imbalance_cost(peak_kw=500, contracted_kw=620)["values"]
        assert values["capacity_surcharge_eur"] == 0.0

    def test_surcharge_uses_the_handbook_rate(self) -> None:
        values = tools.estimate_imbalance_cost(peak_kw=650, contracted_kw=620)["values"]
        assert values["over_kw"] == 30.0
        assert values["capacity_surcharge_eur"] == pytest.approx(30 * 28.0)

    def test_imbalance_component(self) -> None:
        values = tools.estimate_imbalance_cost(
            peak_kw=600, contracted_kw=620, deviation_mwh=2.0
        )["values"]
        assert values["imbalance_cost_eur"] == pytest.approx(180.0)
