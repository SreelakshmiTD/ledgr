import pandas as pd
import pytest

from inject_retries import _compute_injection_probability, _extract_calibration, inject_synthetic_retries

# claude_code (0.2928) x claude-opus-4-5 (relative risk clipped to 2.0) gives
# injection_probability ~0.5856 -- high enough that 150 rows makes the
# probability of getting zero synthetic rows astronomically small
# ((1 - 0.5856)^150 ~ 1e-45), so these tests are deterministic in practice
# despite being based on a probabilistic mechanism.
HARNESS = "claude_code"
MODEL = "claude-opus-4-5"
SAMPLE_SIZE = 150


def test_no_timing_inversions(calls_factory, failure_rates_config):
    df = calls_factory(SAMPLE_SIZE, HARNESS, MODEL)
    original_start = df.set_index("call_id")["start_time"]

    combined = inject_synthetic_retries(df, failure_rates_config, random_seed=42, verbose=False)
    synth = combined[combined["is_synthetic_retry"]]
    assert len(synth) > 0, "expected at least some synthetic rows for this test to be meaningful"

    synth_end = pd.to_datetime(synth["end_time"], format="ISO8601")
    real_orig_start = pd.to_datetime(synth["call_id"].map(original_start), format="ISO8601")
    assert (synth_end < real_orig_start).all()


def test_unmapped_model_raises(calls_factory, failure_rates_config):
    df = calls_factory(5, HARNESS, "totally-unknown-model")
    with pytest.raises(ValueError, match="model_failure_rates"):
        inject_synthetic_retries(df, failure_rates_config, random_seed=42, verbose=False)


def test_unmapped_harness_raises(calls_factory, failure_rates_config):
    df = calls_factory(5, "totally-unknown-harness", MODEL)
    with pytest.raises(ValueError, match="harness_failure_rates"):
        inject_synthetic_retries(df, failure_rates_config, random_seed=42, verbose=False)


def test_reproducibility_is_row_order_independent(calls_factory, failure_rates_config):
    df = calls_factory(SAMPLE_SIZE, HARNESS, MODEL)

    r1 = inject_synthetic_retries(df, failure_rates_config, random_seed=42, verbose=False)
    ids_1 = set(r1.loc[r1["is_synthetic_retry"], "call_id"])

    shuffled = df.sample(frac=1, random_state=7).reset_index(drop=True)
    r2 = inject_synthetic_retries(shuffled, failure_rates_config, random_seed=42, verbose=False)
    ids_2 = set(r2.loc[r2["is_synthetic_retry"], "call_id"])

    assert len(ids_1) > 0
    assert ids_1 == ids_2


def test_relative_risk_clipping_upper_bound(failure_rates_config):
    calibration = _extract_calibration(failure_rates_config)
    harness_rates = failure_rates_config["harness_failure_rates"]
    mean_model_rate = sum(failure_rates_config["model_failure_rates"].values()) / len(
        failure_rates_config["model_failure_rates"]
    )

    # A model with an absurdly high rate relative to the mean -- without
    # clipping this would multiply harness_rate by ~1300x.
    model_rates = dict(failure_rates_config["model_failure_rates"])
    model_rates["extreme-model"] = 100.0
    raw_relative_risk = 100.0 / mean_model_rate

    upper_bound = calibration["relative_risk_bounds"][1]
    assert raw_relative_risk > upper_bound, "test setup should exercise the clip, not stay under it"

    prob = _compute_injection_probability(
        pd.Series(["claude_code"]), pd.Series(["extreme-model"]), harness_rates, model_rates, mean_model_rate, calibration
    )

    max_possible = harness_rates["claude_code"] * upper_bound
    assert prob.iloc[0] == pytest.approx(min(max_possible, calibration["injection_probability_cap"]))


def test_relative_risk_clipping_lower_bound(failure_rates_config):
    """The bound that actually fires in production: gpt-5.2-2025-12-11 (raw
    rate 0.0) and gemini-3-pro-preview (raw relative risk 0.3084) both get
    clipped UP to the lower bound in the real full-dataset run, not left at
    their raw low value."""
    calibration = _extract_calibration(failure_rates_config)
    harness_rates = failure_rates_config["harness_failure_rates"]
    mean_model_rate = sum(failure_rates_config["model_failure_rates"].values()) / len(
        failure_rates_config["model_failure_rates"]
    )

    # A model with a documented rate of exactly 0.0 -- mirrors the real
    # gpt-5.2-2025-12-11 case, not just a hypothetical.
    model_rates = dict(failure_rates_config["model_failure_rates"])
    model_rates["ultra-safe-model"] = 0.0
    raw_relative_risk = 0.0 / mean_model_rate

    lower_bound = calibration["relative_risk_bounds"][0]
    assert raw_relative_risk < lower_bound, "test setup should exercise the clip, not stay above it"

    prob = _compute_injection_probability(
        pd.Series(["claude_code"]), pd.Series(["ultra-safe-model"]), harness_rates, model_rates, mean_model_rate, calibration
    )

    expected = harness_rates["claude_code"] * lower_bound
    assert prob.iloc[0] == pytest.approx(min(expected, calibration["injection_probability_cap"]))
    # explicitly confirm it did NOT stay at the raw (near-zero) value
    assert prob.iloc[0] > raw_relative_risk * harness_rates["claude_code"] + 1e-9


def test_is_synthetic_retry_flag_correctness(calls_factory, failure_rates_config):
    df = calls_factory(SAMPLE_SIZE, HARNESS, MODEL)
    combined = inject_synthetic_retries(df, failure_rates_config, random_seed=42, verbose=False)

    real_rows = combined[~combined["is_synthetic_retry"]]
    synth_rows = combined[combined["is_synthetic_retry"]]

    assert len(synth_rows) > 0
    assert (real_rows["is_synthetic_retry"] == False).all()  # noqa: E712
    assert (synth_rows["is_synthetic_retry"] == True).all()  # noqa: E712
    assert synth_rows["attempt_id"].str.endswith("_attempt_0").all()
    assert real_rows["attempt_id"].str.endswith("_attempt_1").all()


def test_synthetic_output_message_length_is_zero(calls_factory, failure_rates_config):
    """Locks in the fix made after data dictionary review: a synthetic
    failed row must have output_message_length == 0 alongside
    output_tokens == 0 (a failed call produces no output), not inherit a
    nonzero length from its real predecessor."""
    df = calls_factory(SAMPLE_SIZE, HARNESS, MODEL)
    combined = inject_synthetic_retries(df, failure_rates_config, random_seed=42, verbose=False)

    synth = combined[combined["is_synthetic_retry"]]
    assert len(synth) > 0
    assert (synth["output_message_length"] == 0).all()
    assert (synth["output_tokens"] == 0).all()

    # real rows should be untouched -- confirms the fix didn't overreach
    real = combined[~combined["is_synthetic_retry"]]
    assert (real["output_message_length"] == 50).all()  # calls_factory's fixed fixture value


def test_different_seeds_select_different_rows(calls_factory, failure_rates_config):
    """Guards against a future refactor silently ignoring the seed
    argument -- e.g. _stable_uniform() hardcoding a constant instead of
    using its seed parameter would pass every other test in this file
    (they all use seed=42 consistently) but should fail this one."""
    df = calls_factory(SAMPLE_SIZE, HARNESS, MODEL)

    r42 = inject_synthetic_retries(df, failure_rates_config, random_seed=42, verbose=False)
    ids_42 = set(r42.loc[r42["is_synthetic_retry"], "call_id"])

    r43 = inject_synthetic_retries(df, failure_rates_config, random_seed=43, verbose=False)
    ids_43 = set(r43.loc[r43["is_synthetic_retry"], "call_id"])

    assert len(ids_42) > 0
    assert len(ids_43) > 0
    assert ids_42 != ids_43


def test_inject_all_shards_writes_correct_output(tmp_path, monkeypatch, calls_factory):
    """Exercises the actual pipeline entry point end-to-end (file read ->
    inject -> file write), not just the pure inject_synthetic_retries()
    function -- catches bugs in the glob pattern / output path / naming
    that reading pre-existing data/processed/ files would never catch."""
    import inject_retries

    df = calls_factory(SAMPLE_SIZE, HARNESS, MODEL)
    input_path = tmp_path / "shard_0000_calls.parquet"
    df.to_parquet(input_path, index=False)

    monkeypatch.setattr(inject_retries, "PROCESSED_DIR", str(tmp_path))

    output_paths = inject_retries.inject_all_shards()

    expected_output = tmp_path / "shard_0000_augmented.parquet"
    assert str(expected_output) in output_paths
    assert expected_output.exists()

    result = pd.read_parquet(expected_output)
    assert set(result.columns) == set(df.columns) | {"is_synthetic_retry", "error_type"}
    assert len(result) > len(df), "expected some synthetic rows to have been added"
    assert (result["is_synthetic_retry"] == True).sum() > 0  # noqa: E712
    assert (result["is_synthetic_retry"] == False).sum() == len(df)  # noqa: E712
