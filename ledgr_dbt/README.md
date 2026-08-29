# Ledgr dbt Project

## Models
- `mart_cost_per_success`: cost-per-successful-outcome by model+harness
- `mart_cost_anomalies`: daily cost-per-successful-outcome by model, with 
  a 7-day rolling baseline anomaly flag

## Known dependencies and limitations (not yet resolved)

1. **No orchestrated run order.** Both Gold models currently must be run 
   manually (`dbt run`). Neither is triggered automatically after Silver 
   materialization. This is resolved by the Airflow DAG (planned, not yet 
   built), which will trigger `dbt run` as a step after Silver completes.

2. **Cross-mart consistency unverified.** `mart_cost_per_success` (grain: 
   model+harness) and `mart_cost_anomalies` (grain: model+day) both derive 
   cost-per-success from the same Silver source, but nothing currently 
   verifies their aggregate numbers are mutually consistent when rolled up 
   the same way. Planned as part of the final review pass.

3. **Rolling baseline early-window reliability.** The 7-day rolling mean/
   stddev in `mart_cost_anomalies` uses `ROWS BETWEEN 6 PRECEDING AND 1 
   PRECEDING`, which does not distinguish between a window with genuinely 
   7 prior days available versus only 1-2. Early days in each model's 
   timeline have a less statistically reliable baseline. See the model's 
   `min_days_in_window` column (added below) for visibility into this.
