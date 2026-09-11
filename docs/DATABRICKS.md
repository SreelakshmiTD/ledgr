# Ledgr — Databricks / Unity Catalog / dbt

This document covers the Databricks-side portion of Ledgr: Bronze/Silver/
Gold Delta Lake tables, Unity Catalog governance, and dbt models. The local
pandas pipeline (documented in the main README.md) produces intermediate
data that seeds this side; the Databricks side re-implements the core
transformation logic in PySpark and dbt for real distributed-processing
and analytics-engineering depth, cross-validated at every step against the
local pipeline's known-correct output.

## Architecture

Raw HuggingFace parquet files (data/raw/*.parquet locally)
    -> uploaded to S3 (s3://ledgr-raw-data-2026/raw/)
    -> Bronze (ledgr.bronze.sessions_raw): raw session-grain ingestion,
       untouched nested spans column
    -> Silver (ledgr.silver.calls_enriched): exploded to call-grain,
       normalized, synthetic retry injection, cost computed, outcome
       state derived
    -> Gold (ledgr.gold.*): dbt-managed analytics marts

## Catalog structure

- ledgr.bronze.sessions_raw — raw ingestion (10,056 rows, one per session)
- ledgr.bronze.sessions_raw_autoloader — Auto Loader proof-of-concept
  table, demonstrates incremental ingestion (verified: adding a 10th file
  increased row count by exactly that file's row count, not a full reprocess)
- ledgr.silver.calls_enriched — call-grain enriched data (265,311 rows:
  241,473 real + 23,838 synthetic retries)
- ledgr.silver.injection_calibration_audit — audit trail of the retry
  injection calibration math (harness_rate, model_rate, relative_risk,
  injection_probability per call), regenerated as part of every
  materialize_silver() run
- ledgr.gold.mart_cost_per_success — cost-per-successful-outcome by
  model+harness, both variants (including-waste and success-only)
- ledgr.gold.mart_cost_anomalies — daily cost-per-successful-outcome per
  model with a 7-day rolling baseline anomaly flag, rows with fewer than
  3 prior days flagged baseline_unreliable and never marked anomalous
- ledgr.gold.mart_success_outcome_reconciliation — tracks the mismatch
  rate between session-level success and call-level outcome_state,
  excluding synthetic rows (see ADR-0003 and its correction)
- ledgr.gold.sessions_analyst_view — restricted view (7 of 20+ columns)
  demonstrating Unity Catalog column-level governance

## Governance

The `ledgr_analysts` group (created via the account console) is granted
SELECT only on sessions_analyst_view, confirmed via SHOW GRANTS to have
zero access to the underlying tables.

## dbt

Located in ledgr_dbt/. Connects to Databricks via ~/.dbt/profiles.yml
(not committed). Manages the Gold layer only. Run: `cd ledgr_dbt && dbt run && dbt test`

## Known limitations

- MERGE-only-insert write pattern: records are immutable once written.
- Anomaly detection is a baseline statistical demonstration (21 distinct
  days of data), not production-grade.
- No orchestration yet: models are run manually, not triggered by Airflow.
- See ADR-0002 and ADR-0003 for detailed decision rationale.

## Setup note

The dbt connection uses a personal access token (`~/.dbt/profiles.yml`,
not committed) with a 90-day expiration, created 2026-08-29. If dbt
commands start failing with an authentication error after that date, the
token needs to be regenerated in Databricks (Settings > Developer > Access
tokens) and `profiles.yml` updated.
