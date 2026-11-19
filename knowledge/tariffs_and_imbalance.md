# Grid tariffs and imbalance settlement (client handbook extract)

## Contracted capacity and peak charges
Each connection has a contracted transport capacity in kW. The network operator
bills a monthly capacity charge based on the highest 15-minute average power
measured in the billing month. Exceeding the contracted capacity triggers a
surcharge of EUR 28 per kW over contract, applied to the single highest
excursion of the month. A site that crosses its contracted capacity once pays
the surcharge for the whole month, so a single unmanaged peak is expensive.

## Imbalance exposure
Day-ahead nominations are submitted per half-hour. Any deviation between the
nomination and metered offtake is settled at the imbalance price. For planning
purposes the client uses a flat assumption of EUR 90 per MWh of absolute
deviation, which is the trailing twelve-month average. Under-nomination and
over-nomination are both charged; there is no free band.

## Peak shaving priority order
When a forecast peak is expected to exceed 92 percent of contracted capacity,
the site follows a fixed order of measures: first shift flexible HVAC pre-cooling
earlier by up to 90 minutes, second discharge the battery if state of charge is
above 40 percent, third curtail non-critical ventilation zones, and only then
inform the client's energy manager that a capacity excursion is likely.

## Reporting obligations
Any half-hour in which metered demand exceeds 95 percent of contracted capacity
must be reported in the weekly operations note with the measures taken. Forecast
driven interventions are logged with the forecast that triggered them, including
the model version and the error metric at the time of the decision.
