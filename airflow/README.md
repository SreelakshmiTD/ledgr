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

## Known limitations

- **`FERNET_KEY` is not set** (defaults to blank). Acceptable for local
  development only — Airflow uses it to encrypt stored connection
  credentials, so any real deployment would need a real key set.
- **No `email_on_failure` or SLA alerting configured.** Failures are
  currently visible only via the Airflow UI/logs, not pushed to any
  external notification channel.
