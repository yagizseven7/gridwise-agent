"""The classic ML part of the system.

There is a temptation to let an LLM "predict" numbers. It should not. The LLM
decides *which* question to answer and *how* to explain it; a boring, testable,
backtested regression produces the numbers.

Model: ridge regression on calendar features, temperature (with degree-day
transforms) and lagged demand. Evaluated with a walk-forward hold-out against a
seasonal-naive baseline (same half-hour, one week ago), which is the honest
benchmark for half-hourly load.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZON_STEPS = 48  # 24 hours at half-hourly resolution
LAGS = (48, 96, 336)  # 1 day, 2 days, 1 week back


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.clip(np.abs(y_true), 1e-6, None)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a raw time series into a supervised learning table."""
    out = df.copy()
    ts = pd.to_datetime(out["ts"], utc=True)
    hour = ts.dt.hour + ts.dt.minute / 60.0

    out["sin_day"] = np.sin(2 * np.pi * hour / 24.0)
    out["cos_day"] = np.cos(2 * np.pi * hour / 24.0)
    out["sin_week"] = np.sin(2 * np.pi * ts.dt.day_of_week / 7.0)
    out["cos_week"] = np.cos(2 * np.pi * ts.dt.day_of_week / 7.0)
    out["hdd"] = np.clip(15.0 - out["temp_c"], 0, None)
    out["cdd"] = np.clip(out["temp_c"] - 20.0, 0, None)

    for lag in LAGS:
        out[f"lag_{lag}"] = out["demand_kw"].shift(lag)
    out["roll_mean_48"] = out["demand_kw"].shift(1).rolling(48).mean()

    return out.dropna().reset_index(drop=True)


FEATURES = [
    "sin_day", "cos_day", "sin_week", "cos_week",
    "temp_c", "hdd", "cdd", "is_weekend",
    *[f"lag_{lag}" for lag in LAGS],
    "roll_mean_48",
]


@dataclass
class ForecastResult:
    site_id: str
    horizon_hours: int
    peak_kw: float
    peak_at: str
    mean_kw: float
    total_kwh: float
    mape: float
    baseline_mape: float
    improvement_pct: float
    n_train_rows: int
    points: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def forecast(series: pd.DataFrame, site_id: str, horizon_hours: int = 24) -> ForecastResult:
    """Fit on everything before the hold-out window, then predict that window."""
    horizon_steps = max(1, int(horizon_hours * 2))
    table = build_features(series)
    if len(table) < horizon_steps + 400:
        raise ValueError("Not enough history to train a forecaster for this site.")

    train = table.iloc[:-horizon_steps]
    test = table.iloc[-horizon_steps:]

    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(train[FEATURES], train["demand_kw"])
    predictions = np.asarray(model.predict(test[FEATURES]), dtype=float)

    actual = test["demand_kw"].to_numpy(dtype=float)
    baseline = test["lag_336"].to_numpy(dtype=float)  # same slot, one week ago

    model_mape = mape(actual, predictions)
    base_mape = mape(actual, baseline)
    improvement = (base_mape - model_mape) / base_mape * 100 if base_mape else 0.0

    peak_idx = int(np.argmax(predictions))
    timestamps = pd.to_datetime(test["ts"], utc=True).dt.strftime("%Y-%m-%d %H:%M")

    points = [
        {"ts": timestamps.iloc[i], "predicted_kw": round(float(predictions[i]), 1),
         "actual_kw": round(float(actual[i]), 1)}
        for i in range(0, horizon_steps, max(1, horizon_steps // 12))
    ]

    return ForecastResult(
        site_id=site_id,
        horizon_hours=horizon_hours,
        peak_kw=round(float(predictions[peak_idx]), 1),
        peak_at=str(timestamps.iloc[peak_idx]),
        mean_kw=round(float(predictions.mean()), 1),
        total_kwh=round(float(predictions.sum() * 0.5), 1),  # 30-min steps -> kWh
        mape=round(model_mape, 4),
        baseline_mape=round(base_mape, 4),
        improvement_pct=round(improvement, 1),
        n_train_rows=int(len(train)),
        points=points,
    )
