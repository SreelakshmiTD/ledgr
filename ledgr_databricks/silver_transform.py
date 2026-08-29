from pyspark.sql import functions as F, SparkSession

HARNESS_RATES = {
    "claude_code": 0.2928,
    "openai_solo": 0.026,
    "smolagents_code": 0.0005,
    "tool_calling": 0.0023,
    "tool_calling_with_shortlisting": 0.0006,
}
MODEL_RATES = {
    "DeepSeek-V3.2": 0.0574,
    "Kimi-K2.5": 0.0707,
    "claude-opus-4-5": 0.2294,
    "gemini-3-pro-preview": 0.0235,
    "gpt-5.2-2025-12-11": 0.0,
}
MEAN_MODEL_RATE = sum(MODEL_RATES.values()) / len(MODEL_RATES)
SEED = 42


def explode_bronze_sessions(bronze_df):
    """Explode session-level Bronze data into call-level rows."""
    return bronze_df.select(
        "session_id", "run_id", "harness", "benchmark", "benchmark_subset",
        "success", "agent_cost", "execution_time",
        F.posexplode("spans").alias("span_index", "span")
    )


def extract_call_fields(exploded_df):
    """Extract gen_ai.* fields from the nested span struct into flat columns."""
    df = exploded_df.select(
        "session_id", "run_id", "harness", "benchmark", "success",
        "agent_cost", "execution_time",
        F.col("span.span_id").alias("call_id"),
        F.col("span.trace_id").alias("trace_id"),
        F.col("span.parent_span_id").alias("parent_span_id"),
        F.col("span.start_time").alias("start_time"),
        F.col("span.end_time").alias("end_time"),
        F.col("span.status.code").alias("status_code"),
        F.col("span.status.message").alias("status_message"),
        F.col("span.attributes.`gen_ai.request.model`").alias("model_request"),
        F.col("span.attributes.`gen_ai.response.model`").alias("model_response"),
        F.col("span.attributes.`gen_ai.usage.input_tokens`").alias("input_tokens"),
        F.col("span.attributes.`gen_ai.usage.output_tokens`").alias("output_tokens"),
        F.col("span.attributes.`gen_ai.provider.name`").alias("provider"),
        F.length(F.col("span.attributes.`gen_ai.input.messages`")).alias("input_message_length"),
        F.length(F.col("span.attributes.`gen_ai.output.messages`")).alias("output_message_length"),
        F.col("span.attributes.`gen_ai.tool.definitions`").isNotNull().alias("has_tool_definitions"),
    )
    return df.withColumn("task_id", F.col("session_id"))


def validate_config_coverage(df, harness_rates=HARNESS_RATES, model_rates=MODEL_RATES):
    """Fail-loud check: every harness/model in the data must have a configured rate."""
    actual_harnesses = set(row.harness for row in df.select("harness").distinct().collect())
    actual_models = set(row.model_request for row in df.select("model_request").distinct().collect())

    unmapped_harnesses = actual_harnesses - set(harness_rates.keys())
    unmapped_models = actual_models - set(model_rates.keys())

    if unmapped_harnesses:
        raise ValueError(f"harness_rates config is missing rates for: {unmapped_harnesses}")
    if unmapped_models:
        raise ValueError(f"model_rates config is missing rates for: {unmapped_models}")
    return True


def compute_injection_probability(df, harness_rates=HARNESS_RATES, model_rates=MODEL_RATES,
                                     mean_model_rate=MEAN_MODEL_RATE, seed=SEED):
    """Compute per-row synthetic retry injection probability, harness-anchored with clipped model relative risk."""
    harness_map = F.create_map([F.lit(x) for pair in harness_rates.items() for x in pair])
    model_map = F.create_map([F.lit(x) for pair in model_rates.items() for x in pair])

    return (df
        .withColumn("attempt_id", F.concat(F.col("call_id"), F.lit("_attempt_1")))
        .withColumn("harness_rate", harness_map[F.col("harness")])
        .withColumn("model_rate", model_map[F.col("model_request")])
        .withColumn("relative_risk_raw", F.col("model_rate") / F.lit(mean_model_rate))
        .withColumn("relative_risk", F.least(F.greatest(F.col("relative_risk_raw"), F.lit(0.5)), F.lit(2.0)))
        .withColumn("injection_probability", F.least(F.col("harness_rate") * F.col("relative_risk"), F.lit(0.95)))
        .withColumn("hash_uniform", (F.pmod(F.xxhash64(F.col("call_id"), F.lit(seed)), F.lit(1000000)) / F.lit(1000000.0)))
        .withColumn("is_selected_for_injection", F.col("hash_uniform") < F.col("injection_probability"))
    )

PRICING = {
    "DeepSeek-V3.2": {"input_per_million": 0.580, "output_per_million": 1.68},
    "Kimi-K2.5": {"input_per_million": 0.60, "output_per_million": 3.00},
    "claude-opus-4-5": {"input_per_million": 5.00, "output_per_million": 25.00},
    "gemini-3-pro-preview": {"input_per_million": 2.00, "output_per_million": 12.00},
    "gpt-5.2-2025-12-11": {"input_per_million": 1.75, "output_per_million": 14.00},
}


def validate_pricing_coverage(df, pricing=PRICING):
    """Fail-loud check: every model in the data must have pricing configured."""
    actual_models = set(row.model_request for row in df.select("model_request").distinct().collect())
    unmapped = actual_models - set(pricing.keys())
    if unmapped:
        raise ValueError(f"pricing config is missing rates for: {unmapped}")
    return True


def compute_execution_cost(df, pricing=PRICING):
    """Calculate execution_cost_usd per row based on model and token counts."""
    input_price_map = F.create_map([
        F.lit(x) for model, rates in pricing.items() for x in (model, rates["input_per_million"])
    ])
    output_price_map = F.create_map([
        F.lit(x) for model, rates in pricing.items() for x in (model, rates["output_per_million"])
    ])
    return df.withColumn(
        "execution_cost_usd",
        (F.col("input_tokens") / F.lit(1_000_000) * input_price_map[F.col("model_request")]) +
        (F.col("output_tokens") / F.lit(1_000_000) * output_price_map[F.col("model_request")])
    )
def generate_synthetic_retries(df, seed=SEED):
    selected_df = df.filter(F.col("is_selected_for_injection") == True)

    synthetic_df = (selected_df
        .withColumn("synth_duration_seed", F.pmod(F.xxhash64(F.col("call_id"), F.lit(seed + 1)), F.lit(4000)))
        .withColumn("retry_duration_sec", (F.col("synth_duration_seed") / F.lit(1000.0)) + F.lit(1.0))
        .withColumn("synth_buffer_seed", F.pmod(F.xxhash64(F.col("call_id"), F.lit(seed + 2)), F.lit(9000)))
        .withColumn("buffer_sec", (F.col("synth_buffer_seed") / F.lit(1000.0)) + F.lit(1.0))
        .withColumn("pre_offset_sec", F.col("retry_duration_sec") + F.col("buffer_sec"))
        .withColumn("real_start_ts", F.to_timestamp(F.col("start_time"), "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX"))
        .withColumn("synth_start_time", F.col("real_start_ts") - F.expr("INTERVAL 1 SECONDS") * F.col("pre_offset_sec"))
        .withColumn("synth_end_time", F.col("synth_start_time") + F.expr("INTERVAL 1 SECONDS") * F.col("retry_duration_sec"))
    )
def add_outcome_state(df):
    """
    Derive an outcome_state column from status_code and is_synthetic_retry.
    - Synthetic (failed) attempts are always FAILED
    - Real attempts with status_code == 1 (OTel success convention) are SUCCESS
    - Real attempts with any other status_code are FAILED
    - EVALUATION_ERROR is reserved for a future evaluation layer, not derivable
      from current data, so it never appears here, documented as a known gap
    """
    return df.withColumn(
        "outcome_state",
        F.when(F.col("is_synthetic_retry") == True, F.lit("FAILED"))
         .when(F.col("status_code") == 1, F.lit("SUCCESS"))
         .otherwise(F.lit("FAILED"))
    )
    
def validate_synthetic_retries(synthetic_df, real_df):
    """
    Fail-loud check: synthetic retry rows must have no null timestamps and 
    must never have an end_time at or after the corresponding real row's 
    original start_time (the core timing-inversion guarantee).
    """
    null_timestamps = synthetic_df.filter(
        F.col("start_time").isNull() | F.col("end_time").isNull()
    ).count()
    if null_timestamps > 0:
        raise ValueError("Found " + str(null_timestamps) + " synthetic rows with null timestamps")

    check_df = synthetic_df.join(
        real_df.select("call_id", F.col("start_time").alias("real_start_time")),
        on="call_id"
    )
    inversions = check_df.filter(
        F.to_timestamp(F.col("end_time")) >= F.to_timestamp(F.col("real_start_time"))
    ).count()
    if inversions > 0:
        raise ValueError("Found " + str(inversions) + " timing inversions in synthetic rows")

    return True
    result = synthetic_df.select(
        F.col("task_id"), F.col("run_id"), F.col("trace_id"),
        F.col("call_id"),
        F.concat(F.col("call_id"), F.lit("_attempt_0")).alias("attempt_id"),
        F.col("synth_start_time").cast("string").alias("start_time"),
        F.col("synth_end_time").cast("string").alias("end_time"),
        F.lit(2).alias("status_code"),
        F.lit("synthetic_failure").alias("status_message"),
        F.col("model_request"), F.col("model_response"),
        F.col("input_tokens"),
        F.lit(0).alias("output_tokens"),
        F.col("provider"),
        F.col("input_message_length"),
        F.lit(0).alias("output_message_length"),
        F.col("has_tool_definitions"),
        F.col("harness"), F.col("benchmark"), F.col("success"),
        F.col("execution_cost_usd"),
        F.lit(True).alias("is_synthetic_retry"),
    )
    return result