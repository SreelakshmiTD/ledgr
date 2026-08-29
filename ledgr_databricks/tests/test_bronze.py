import pytest
from pyspark.sql import SparkSession, functions as F


@pytest.fixture(scope="module")
def spark():
    return SparkSession.builder.getOrCreate()


def test_bronze_table_has_expected_schema_and_grain(spark):
    """
    Confirm Bronze ingestion produces the expected raw session-grain schema:
    one row per session, nested spans column intact, no data loss vs. source.
    """
    bronze_df = spark.table("ledgr.bronze.sessions_raw")

    expected_columns = {
        "schema_version", "config_path", "run_id", "session_id", "harness",
        "benchmark", "benchmark_subset", "models", "score", "success",
        "status", "steps", "action_count", "agent_cost", "benchmark_cost",
        "execution_time", "total_tokens", "max_tokens", "spans", "collected_at"
    }
    actual_columns = set(bronze_df.columns)
    assert actual_columns == expected_columns

    row_count = bronze_df.count()
    assert row_count == 10056

    distinct_sessions = bronze_df.select("session_id").distinct().count()
    assert distinct_sessions == row_count, "Bronze should have exactly one row per session_id, no duplicates"

    null_session_ids = bronze_df.filter(F.col("session_id").isNull()).count()
    assert null_session_ids == 0