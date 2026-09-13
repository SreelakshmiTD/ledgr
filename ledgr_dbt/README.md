# Ledgr dbt Project

## Models
- `mart_cost_per_success`: cost-per-successful-outcome by model+harness
- `mart_cost_anomalies`: daily cost-per-successful-outcome by model, with
  a 7-day rolling baseline anomaly flag
- `mart_success_outcome_reconciliation`: mismatch rate between
  session-level success and call-level outcome_state, excluding
  synthetic rows (see ADR-0003 and its correction)

## Orchestration

Triggered by the Airflow DAG (`ledgr_pipeline`, see `airflow/README.md`)
as the final step after Silver materialization completes. Can also be
run manually: `cd ledgr_dbt && dbt run && dbt test`.

## Verified

Cross-mart consistency: `mart_cost_per_success` (grain: model+harness)
and `mart_cost_anomalies` (grain: model+day) both derive cost-per-success
from the same Silver source. Gold/Silver cost aggregation reconciles
exactly: $33,651.89 total, matching to within floating-point precision.

## Known limitations

**Rolling baseline early-window reliability.** The 7-day rolling mean/
stddev in `mart_cost_anomalies` uses `ROWS BETWEEN 6 PRECEDING AND 1
PRECEDING`, which does not distinguish between a window with genuinely
7 prior days available versus only 1-2. Early days in each model's
timeline have a less statistically reliable baseline. See the model's
`days_in_window` column for visibility into this.
