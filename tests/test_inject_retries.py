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


def test_relative_risk_clipping(failure_rates_config):
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
