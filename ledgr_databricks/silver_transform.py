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