from datetime import datetime
from airflow import DAG
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "ledgr",
    "retries": 1,
}

with DAG(
    dag_id="ledgr_pipeline",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ledgr"],
) as dag:

    bronze_ingestion = DatabricksRunNowOperator(
        task_id="bronze_ingestion",
        databricks_conn_id="databricks_default",
        job_id=106794160105586,
    )

    silver_materialize = DatabricksRunNowOperator(
        task_id="silver_materialize",
        databricks_conn_id="databricks_default",
        job_id=736829007447126,
    )

    dbt_run_and_test = BashOperator(
        task_id="dbt_run_and_test",
        bash_command="cd /opt/ledgr/ledgr_dbt && dbt run && dbt test",
    )

    bronze_ingestion >> silver_materialize >> dbt_run_and_test
