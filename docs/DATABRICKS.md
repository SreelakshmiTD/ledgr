# Databricks / Unity Catalog / dbt implementation

Ledgr also has a Databricks/Unity Catalog implementation of the Silver and
Gold layers, built on top of the Bronze table produced by uploading this
local pipeline's `data/processed/shard_*_final.parquet` output to S3. This
is a separate environment/codebase from the local pandas pipeline in
`src/` — see `docs/adr/0002-bronze-constraint-scope.md` and
`docs/adr/0003-success-vs-outcome-state-reconciliation.md` for the
architectural decisions and findings documented from that side.

- `ledgr_databricks/` — PySpark Silver transform (`silver_transform.py`)
  that explodes Bronze session rows into call-level rows, validates
  harness/model config coverage, computes synthetic retry injection and
  cost, and materializes `ledgr.silver.calls_enriched`. Notebooks
  (`01_bronze_ingestion.ipynb`, `02_silver_normalization.ipynb`,
  `ledgr_databricks/03_silver_materialize.ipynb`,
  `ledgr_databricks/03_gold_marts.ipynb`) drive this interactively; pytest
  coverage lives in `ledgr_databricks/tests/`.
- `ledgr_dbt/` — dbt project building the Gold marts
  (`mart_cost_per_success`, `mart_cost_anomalies`,
  `mart_success_outcome_reconciliation`) on top of the Silver table, with
  dbt tests asserting business invariants (e.g. `successful_calls <=
  total_calls`, non-negative wasted-retry cost).

## Setup note

The dbt connection uses a personal access token (`~/.dbt/profiles.yml`,
not committed) with a 90-day expiration, created 2026-08-29. If dbt
commands start failing with an authentication error after that date, the
token needs to be regenerated in Databricks (Settings > Developer > Access
tokens) and `profiles.yml` updated.
