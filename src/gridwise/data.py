"""The data layer.

The agent reads real rows out of a real analytical database (DuckDB). The data
itself is synthetic, but it is generated with the shape of genuine half-hourly
building telemetry: a daily profile, a weekly profile, weather dependence, a
slow efficiency trend and noise.

Swapping DuckDB for Snowflake / BigQuery / Athena means changing this file only.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .config import settings

SITES = {
    "NL-UTR-01": {"base_kw": 420.0, "hdd_sensitivity": 11.0, "cdd_sensitivity": 16.0},
    "NL-AMS-02": {"base_kw": 260.0, "hdd_sensitivity": 7.5, "cdd_sensitivity": 12.0},
}

FORBIDDEN_SQL = (
    "insert", "update", "delete", "drop", "alter", "create", "attach",
    "copy", "export", "install", "load", "pragma", "set", "reset", "checkpoint",
)


def generate_readings(days: int = 400, seed: int = 7) -> pd.DataFrame:
    """Build a half-hourly telemetry table for every site."""
    rng = np.random.default_rng(seed)
    periods = days * 48
    index = pd.date_range(end=pd.Timestamp("2026-09-01", tz="UTC"), periods=periods, freq="30min")

    frames = []
    for site_id, cfg in SITES.items():
        hour = index.hour + index.minute / 60.0
        dow = index.day_of_week.to_numpy()
        doy = index.day_of_year.to_numpy()

        # Outdoor temperature: seasonal sine + daily swing + noise (degrees C).
        seasonal = 10.5 + 8.0 * np.sin(2 * np.pi * (doy - 110) / 365.25)
        daily = 3.0 * np.sin(2 * np.pi * (hour - 9) / 24.0)
        temp_c = seasonal + daily + rng.normal(0, 1.2, periods)

        # Occupancy shape: office hours on weekdays, a low floor at night.
        office = np.exp(-0.5 * ((hour - 13.0) / 3.4) ** 2)
        weekday = np.where(dow < 5, 1.0, 0.28)
        occupancy = 0.18 + 0.82 * office * weekday

        # Heating below 15C, cooling above 20C.
        hdd = np.clip(15.0 - temp_c, 0, None)
        cdd = np.clip(temp_c - 20.0, 0, None)

        efficiency_trend = np.linspace(1.04, 0.97, periods)  # retrofit rolling out
        demand = (
            cfg["base_kw"] * occupancy
            + cfg["hdd_sensitivity"] * hdd * (0.35 + 0.65 * occupancy)
            + cfg["cdd_sensitivity"] * cdd * (0.30 + 0.70 * occupancy)
        ) * efficiency_trend
        demand = np.clip(demand + rng.normal(0, 9.0, periods), 12.0, None)

        # On-site PV: daylight only, weather-dampened.
        solar = np.clip(np.sin(np.pi * (hour - 6.5) / 11.5), 0, None) ** 1.4
        pv_kw = 140.0 * solar * rng.uniform(0.45, 1.0, periods)

        frames.append(
            pd.DataFrame(
                {
                    "ts": index,
                    "site_id": site_id,
                    "demand_kw": np.round(demand, 2),
                    "pv_kw": np.round(pv_kw, 2),
                    "temp_c": np.round(temp_c, 2),
                    "is_weekend": (dow >= 5).astype(int),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def build_database(db_path: Path | None = None, days: int = 400, seed: int = 7) -> Path:
    """(Re)build the DuckDB file. This is the tiny 'ETL' step of the demo."""
    path = Path(db_path or settings.db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    readings = generate_readings(days=days, seed=seed)

    con = duckdb.connect(str(path))
    try:
        con.execute("DROP TABLE IF EXISTS readings")
        con.execute(
            """
            CREATE TABLE readings (
                ts TIMESTAMPTZ,
                site_id VARCHAR,
                demand_kw DOUBLE,
                pv_kw DOUBLE,
                temp_c DOUBLE,
                is_weekend INTEGER
            )
            """
        )
        con.register("readings_df", readings)
        con.execute("INSERT INTO readings SELECT * FROM readings_df")
        con.execute("CREATE INDEX IF NOT EXISTS idx_readings_site_ts ON readings (site_id, ts)")
    finally:
        con.close()
    return path


def ensure_database(db_path: Path | None = None) -> Path:
    path = Path(db_path or settings.db_path)
    if not path.exists():
        build_database(path)
    return path


def _connect_read_only(path: Path) -> duckdb.DuckDBPyConnection:
    """Open the warehouse read-only with file-system access switched off.

    `read_only=True` only protects the database file. DuckDB can also read any
    file on disk (`read_text('/etc/passwd')`, `FROM 'secrets.csv'`), so external
    access is disabled at connection time. It cannot be re-enabled from SQL.
    """
    return duckdb.connect(str(path), read_only=True, config={"enable_external_access": False})


def validate_site(site_id: str) -> str:
    if site_id not in SITES:
        raise ValueError(f"Unknown site '{site_id}'. Known sites: {sorted(SITES)}")
    return site_id


def is_read_only(sql: str) -> bool:
    """First line of defence: one statement, and it must be a SELECT.

    The second line is `_connect_read_only`, which is what actually enforces it.
    """
    cleaned = " ".join(sql.strip().lower().split())
    if not cleaned.startswith(("select", "with")):
        return False
    if ";" in cleaned.rstrip(";"):
        return False
    return not any(f" {word} " in f" {cleaned} " for word in FORBIDDEN_SQL)


def run_sql(sql: str, db_path: Path | None = None, max_rows: int = 200) -> pd.DataFrame:
    """Execute a read-only query and cap the number of rows that come back."""
    if not is_read_only(sql):
        raise PermissionError("Only single read-only SELECT statements are allowed.")
    path = ensure_database(db_path)
    con = _connect_read_only(path)
    try:
        return con.execute(sql).fetch_df().head(max_rows)
    finally:
        con.close()


def load_site_series(site_id: str, db_path: Path | None = None) -> pd.DataFrame:
    validate_site(site_id)
    path = ensure_database(db_path)
    con = _connect_read_only(path)
    try:
        return con.execute(
            "SELECT ts, demand_kw, pv_kw, temp_c, is_weekend FROM readings "
            "WHERE site_id = ? ORDER BY ts",
            [site_id],
        ).fetch_df()
    finally:
        con.close()
