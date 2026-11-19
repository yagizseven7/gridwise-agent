# Model card: half-hourly demand forecaster v0.1

## Intended use
Short-horizon (up to 48 hours) forecasting of building electricity demand in kW
for capacity management and day-ahead nomination support. Not intended for
settlement, billing, or safety-critical control.

## Method and features
Ridge regression on standardised features: sine and cosine encodings of
time-of-day and day-of-week, outdoor temperature with heating and cooling degree
day transforms, a weekend flag, lagged demand at 24 hours, 48 hours and 7 days,
and a 24-hour rolling mean. Trained walk-forward, evaluated on the most recent
held-out window.

## Baseline and acceptance criteria
The reference baseline is seasonal naive: the value from the same half-hour one
week earlier. The model is only promoted if it beats that baseline on mean
absolute percentage error on the hold-out window. A MAPE above 12 percent on the
hold-out window is treated as a degraded model and answers derived from it are
flagged for human review.

## Known limitations
The model has no holiday calendar and will overestimate demand on public
holidays that fall on weekdays. It assumes the weather input is accurate, so
forecast weather error propagates directly. Behaviour during extended building
closures is untested. Retraining cadence in production is weekly, with drift
monitored daily on rolling MAPE.
