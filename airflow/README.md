# Ledgr Airflow orchestration

Orchestrates the Databricks/dbt Silver and Gold materialization
(`ledgr_pipeline` DAG) instead of running notebooks/dbt manually.

## How to run

```bash
docker compose up -d
```

Then open the Airflow UI at [localhost:8080](http://localhost:8080) and
trigger the `ledgr_pipeline` DAG.

## Required setup

A Databricks connection named `databricks_default` must be configured in
Airflow's UI (Admin > Connections) with a personal access token scoped to
`sql`, `clusters`, and `jobs`.

`LEDGR_PROJECT_DIR` and `LEDGR_DBT_PROFILES_DIR` environment variables can
be set (e.g. in a local `.env`) to override the default local paths in
`docker-compose.yaml` — needed on any machine other than the one those
defaults were hardcoded for.

## Databricks Job Setup

The DAG references two existing Databricks Jobs by ID (bronze_ingestion:
job_id=106794160105586, silver_materialize: job_id=736829007447126).
These must be created manually in the target Databricks workspace before
the DAG can run:

1. In Databricks, go to Jobs & Pipelines > Create > Job
2. Create 'ledgr_bronze_ingestion': task type Notebook, path
   ledgr_databricks/01_bronze_ingestion, cluster Serverless
3. Create 'ledgr_silver_materialize': task type Notebook, path
   ledgr_databricks/03_silver_materialize, cluster Serverless
4. Get each job's ID from the Databricks UI (visible in the job's URL
   or details panel) and update the job_id values in
   airflow/dags/ledgr_pipeline_dag.py to match

Note: these job IDs are specific to this project's Databricks workspace
and will NOT work on a different workspace without recreating the jobs
and updating the IDs.

## Known limitations

- **`FERNET_KEY` is not set** (defaults to blank). Acceptable for local
  development only — Airflow uses it to encrypt stored connection
  credentials, so any real deployment would need a real key set.
- **No `email_on_failure` or SLA alerting configured.** Failures are
  currently visible only via the Airflow UI/logs, not pushed to any
  external notification channel.
