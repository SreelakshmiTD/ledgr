import pytest
from pyspark.sql import SparkSession, Row
from databricks.silver_transform import (
    validate_config_coverage,
    compute_injection_probability,
    HARNESS_RATES,
    MODEL_RATES,
)


@pytest.fixture(scope="module")
def spark():
    return SparkSession.builder.appName("ledgr-tests").getOrCreate()


def test_validate_config_coverage_passes_for_known_values(spark):
    df = spark.createDataFrame([
        Row(harness="claude_code", model_request="DeepSeek-V3.2"),
    ])
    assert validate_config_coverage(df) is True


def test_validate_config_coverage_raises_for_unmapped_harness(spark):
    df = spark.createDataFrame([
        Row(harness="totally_unknown_harness", model_request="DeepSeek-V3.2"),
    ])
    with pytest.raises(ValueError, match="harness_rates"):
        validate_config_coverage(df)


def test_validate_config_coverage_raises_for_unmapped_model(spark):
    df = spark.createDataFrame([
        Row(harness="claude_code", model_request="totally_unknown_model"),
    ])
    with pytest.raises(ValueError, match="model_rates"):
        validate_config_coverage(df)


def test_injection_probability_relative_risk_upper_clip(spark):
    df = spark.createDataFrame([
        Row(call_id="c1", harness="claude_code", model_request="claude-opus-4-5"),
    ])
    result = compute_injection_probability(df).collect()[0]
    assert result.relative_risk == 2.0


def test_injection_probability_relative_risk_lower_clip(spark):
    df = spark.createDataFrame([
        Row(call_id="c1", harness="claude_code", model_request="gpt-5.2-2025-12-11"),
    ])
    result = compute_injection_probability(df).collect()[0]
    assert result.relative_risk == 0.5


def test_injection_probability_is_deterministic(spark):
    df = spark.createDataFrame([
        Row(call_id="fixed_call_id_123", harness="claude_code", model_request="DeepSeek-V3.2"),
    ])
    result1 = compute_injection_probability(df).collect()[0].hash_uniform
    result2 = compute_injection_probability(df).collect()[0].hash_uniform
    assert result1 == result2