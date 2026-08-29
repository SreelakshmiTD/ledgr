import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F

from ledgr_databricks.silver_transform import (
    validate_config_coverage,
    compute_injection_probability,
    explode_bronze_sessions,
    extract_call_fields,
    validate_pricing_coverage,
    compute_execution_cost,
    generate_synthetic_retries,
    validate_synthetic_retries,
    add_outcome_state,
    HARNESS_RATES,
    MODEL_RATES,
)


@pytest.fixture(scope="module")
def spark():
    return SparkSession.builder.getOrCreate()


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


def test_explode_bronze_sessions_produces_one_row_per_span(spark):
    bronze_schema = spark.table("ledgr.bronze.sessions_raw").schema

    span_attrs_1 = Row(**{
        "error.type": None, "gen_ai.conversation.id": None,
        "gen_ai.input.messages": "hi", "gen_ai.operation.name": "chat",
        "gen_ai.output.messages": "hello", "gen_ai.output.type": None,
        "gen_ai.provider.name": "anthropic",
        "gen_ai.request.max_tokens": 100, "gen_ai.request.model": "claude-opus-4-5",
        "gen_ai.request.stop_sequences": [], "gen_ai.request.temperature": 0.5,
        "gen_ai.response.finish_reasons": ["stop"], "gen_ai.response.id": "span1",
        "gen_ai.response.model": "claude-opus-4-5", "gen_ai.system_instructions": None,
        "gen_ai.tool.definitions": None,
        "gen_ai.usage.input_tokens": 10, "gen_ai.usage.output_tokens": 5,
    })
    span_attrs_2 = Row(**{
        "error.type": None, "gen_ai.conversation.id": None,
        "gen_ai.input.messages": "hi2", "gen_ai.operation.name": "chat",
        "gen_ai.output.messages": "hello2", "gen_ai.output.type": None,
        "gen_ai.provider.name": "anthropic",
        "gen_ai.request.max_tokens": 100, "gen_ai.request.model": "claude-opus-4-5",
        "gen_ai.request.stop_sequences": [], "gen_ai.request.temperature": 0.5,
        "gen_ai.response.finish_reasons": ["stop"], "gen_ai.response.id": "span2",
        "gen_ai.response.model": "claude-opus-4-5", "gen_ai.system_instructions": None,
        "gen_ai.tool.definitions": None,
        "gen_ai.usage.input_tokens": 20, "gen_ai.usage.output_tokens": 8,
    })
    res_attrs = Row(**{
        "deployment.environment.name": None, "service.name": None,
        "service.namespace": None, "service.version": None,
        "telemetry.sdk.language": None, "telemetry.sdk.name": None,
        "telemetry.sdk.version": None,
    })

    spans_list = [
        Row(span_id="span1", trace_id="t1", parent_span_id=None, name="chat", kind="internal",
            start_time="2026-01-01T00:00:00Z", end_time="2026-01-01T00:00:01Z",
            status=Row(code=1, message=None), attributes=span_attrs_1,
            resource_attributes=res_attrs, events=None),
        Row(span_id="span2", trace_id="t1", parent_span_id=None, name="chat", kind="internal",
            start_time="2026-01-01T00:00:02Z", end_time="2026-01-01T00:00:03Z",
            status=Row(code=1, message=None), attributes=span_attrs_2,
            resource_attributes=res_attrs, events=None),
    ]

    bronze_df = spark.createDataFrame([
        Row(schema_version="1.0", config_path=None, run_id="r1", session_id="s1",
            harness="claude_code", benchmark="test", benchmark_subset=None,
            models=["test"], score=0.0, success=True, status="ok", steps=1,
            action_count=1, agent_cost=1.0, benchmark_cost=0.0, execution_time=10.0,
            total_tokens=15, max_tokens=100, spans=spans_list, collected_at="2026-01-01")
    ], schema=bronze_schema)

    exploded = explode_bronze_sessions(bronze_df)
    assert exploded.count() == 2


def test_extract_call_fields_pulls_correct_values(spark):
    bronze_schema = spark.table("ledgr.bronze.sessions_raw").schema

    span_attrs = Row(**{
        "error.type": None, "gen_ai.conversation.id": None,
        "gen_ai.input.messages": "hi", "gen_ai.operation.name": "chat",
        "gen_ai.output.messages": "hello", "gen_ai.output.type": None,
        "gen_ai.provider.name": "anthropic",
        "gen_ai.request.max_tokens": 100, "gen_ai.request.model": "claude-opus-4-5",
        "gen_ai.request.stop_sequences": [], "gen_ai.request.temperature": 0.5,
        "gen_ai.response.finish_reasons": ["stop"], "gen_ai.response.id": "span1",
        "gen_ai.response.model": "claude-opus-4-5", "gen_ai.system_instructions": None,
        "gen_ai.tool.definitions": None,
        "gen_ai.usage.input_tokens": 42, "gen_ai.usage.output_tokens": 17,
    })
    res_attrs = Row(**{
        "deployment.environment.name": None, "service.name": None,
        "service.namespace": None, "service.version": None,
        "telemetry.sdk.language": None, "telemetry.sdk.name": None,
        "telemetry.sdk.version": None,
    })

    spans_list = [
        Row(span_id="span1", trace_id="t1", parent_span_id=None, name="chat", kind="internal",
            start_time="2026-01-01T00:00:00Z", end_time="2026-01-01T00:00:01Z",
            status=Row(code=1, message=None), attributes=span_attrs,
            resource_attributes=res_attrs, events=None),
    ]

    bronze_df = spark.createDataFrame([
        Row(schema_version="1.0", config_path=None, run_id="r1", session_id="s1",
            harness="claude_code", benchmark="test", benchmark_subset=None,
            models=["test"], score=0.0, success=True, status="ok", steps=1,
            action_count=1, agent_cost=1.0, benchmark_cost=0.0, execution_time=10.0,
            total_tokens=15, max_tokens=100, spans=spans_list, collected_at="2026-01-01")
    ], schema=bronze_schema)

    exploded = explode_bronze_sessions(bronze_df)
    result = extract_call_fields(exploded).collect()[0]
    assert result.call_id == "span1"
    assert result.model_request == "claude-opus-4-5"
    assert result.input_tokens == 42
    assert result.output_tokens == 17
    assert result.provider == "anthropic"
    assert result.task_id == "s1"

def test_compute_execution_cost_basic(spark):
    df = spark.createDataFrame([
        Row(model_request="claude-opus-4-5", input_tokens=1000000, output_tokens=0),
    ])
    result = compute_execution_cost(df).collect()[0]
    assert abs(result.execution_cost_usd - 5.00) < 0.001


def test_validate_pricing_coverage_raises_for_unmapped_model(spark):
    df = spark.createDataFrame([
        Row(model_request="totally_unknown_model"),
    ])
    with pytest.raises(ValueError, match="pricing"):
        validate_pricing_coverage(df)

def test_generate_synthetic_retries_produces_correct_structure(spark):
    bronze_schema = spark.table("ledgr.bronze.sessions_raw").schema

    span_attrs = Row(**{
        "error.type": None, "gen_ai.conversation.id": None,
        "gen_ai.input.messages": "hi", "gen_ai.operation.name": "chat",
        "gen_ai.output.messages": "hello", "gen_ai.output.type": None,
        "gen_ai.provider.name": "anthropic",
        "gen_ai.request.max_tokens": 100, "gen_ai.request.model": "claude-opus-4-5",
        "gen_ai.request.stop_sequences": [], "gen_ai.request.temperature": 0.5,
        "gen_ai.response.finish_reasons": ["stop"], "gen_ai.response.id": "span1",
        "gen_ai.response.model": "claude-opus-4-5", "gen_ai.system_instructions": None,
        "gen_ai.tool.definitions": None,
        "gen_ai.usage.input_tokens": 100, "gen_ai.usage.output_tokens": 50,
    })
    res_attrs = Row(**{
        "deployment.environment.name": None, "service.name": None,
        "service.namespace": None, "service.version": None,
        "telemetry.sdk.language": None, "telemetry.sdk.name": None,
        "telemetry.sdk.version": None,
    })
    spans_list = [
        Row(span_id="span1", trace_id="t1", parent_span_id=None, name="chat", kind="internal",
            start_time="2026-01-01T00:00:10.000000+00:00", end_time="2026-01-01T00:00:11.000000+00:00",
            status=Row(code=1, message=None), attributes=span_attrs,
            resource_attributes=res_attrs, events=None),
    ]
    bronze_df = spark.createDataFrame([
        Row(schema_version="1.0", config_path=None, run_id="r1", session_id="s1",
            harness="claude_code", benchmark="test", benchmark_subset=None,
            models=["test"], score=0.0, success=True, status="ok", steps=1,
            action_count=1, agent_cost=1.0, benchmark_cost=0.0, execution_time=10.0,
            total_tokens=150, max_tokens=100, spans=spans_list, collected_at="2026-01-01")
    ], schema=bronze_schema)

    exploded = explode_bronze_sessions(bronze_df)
    normalized = extract_call_fields(exploded)
    injected = compute_injection_probability(normalized)
    priced = compute_execution_cost(injected)  # <-- this was missing
    forced_df = priced.withColumn("is_selected_for_injection", F.lit(True))

    synthetic = generate_synthetic_retries(forced_df)
    result = synthetic.collect()[0]

    assert result.attempt_id.endswith("_attempt_0")
    assert result.output_tokens == 0
    assert result.output_message_length == 0
    assert result.is_synthetic_retry == True
    assert result.status_code == 2


def test_validate_synthetic_retries_passes_for_valid_data(spark):
    real_df = spark.createDataFrame([
        Row(call_id="c1", start_time="2026-01-01T00:00:10.000000+00:00"),
    ])
    synthetic_df = spark.createDataFrame([
        Row(call_id="c1", start_time="2026-01-01T00:00:00.000000+00:00",
            end_time="2026-01-01T00:00:05.000000+00:00"),
    ])
    assert validate_synthetic_retries(synthetic_df, real_df) is True


def test_validate_synthetic_retries_raises_on_timing_inversion(spark):
    real_df = spark.createDataFrame([
        Row(call_id="c1", start_time="2026-01-01T00:00:10.000000+00:00"),
    ])
    # Bad: synthetic end_time is AFTER real start_time, structurally invalid
    synthetic_df = spark.createDataFrame([
        Row(call_id="c1", start_time="2026-01-01T00:00:00.000000+00:00",
            end_time="2026-01-01T00:00:15.000000+00:00"),
    ])
    with pytest.raises(ValueError, match="timing inversions"):
        validate_synthetic_retries(synthetic_df, real_df)


def test_add_outcome_state_synthetic_always_failed(spark):
    df = spark.createDataFrame([
        Row(status_code=1, is_synthetic_retry=True),
        Row(status_code=1, is_synthetic_retry=False),
        Row(status_code=2, is_synthetic_retry=False),
    ])
    result = add_outcome_state(df).collect()
    states = {(r.is_synthetic_retry, r.status_code): r.outcome_state for r in result}
    assert states[(True, 1)] == "FAILED"
    assert states[(False, 1)] == "SUCCESS"
    assert states[(False, 2)] == "FAILED"