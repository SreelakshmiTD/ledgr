import os

import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "failure_rates.yaml")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

ERROR_TYPES = ["rate_limit", "timeout", "malformed_response"]

# OpenTelemetry status codes seen in the real data: 0=UNSET, 1=OK, 2=ERROR.
# 2 is used below for synthetic failures because it's the convention this
# dataset's own spans actually use (confirmed by inspection, not assumed).
SYNTHETIC_STATUS_CODE = 2

DRIFT_WARNING_THRESHOLD = 0.05  # 5 percentage points
INJECTION_PROBABILITY_CAP = 0.95
RELATIVE_RISK_BOUNDS = (0.5, 2.0)


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _require_mapped(real, harness_rates, model_rates):
    """
    Fail loud on any harness/model not present in the config, instead of
    silently defaulting via fillna(). A silent default can't distinguish "this
    model is documented as very safe" from "we have no idea what this model
    is" -- both used to collapse to the same relative_risk. Better to crash
    with the specific missing name and force a config update than to
    misrepresent an unknown model's risk.
    """
    unmapped_harness = set(real["harness"].unique()) - set(harness_rates.keys())
    if unmapped_harness:
        raise ValueError(
            f"harness_failure_rates in config/failure_rates.yaml is missing a rate "
            f"for: {sorted(unmapped_harness)}"
        )

    if real["model"].isna().any():
        raise ValueError("Found null 'model' values with no known failure rate.")

    unmapped_model = set(real["model"].unique()) - set(model_rates.keys())
    if unmapped_model:
        raise ValueError(
            f"model_failure_rates in config/failure_rates.yaml is missing a rate "
            f"for: {sorted(unmapped_model)}"
        )


def inject_synthetic_retries(df, failure_rates_config, random_seed=42):
    """
    Takes the flattened call-level DataFrame and injects synthetic failed-retry
    spans, calibrated jointly on harness AND model, not just harness alone.
    """
    rng = np.random.default_rng(random_seed)

    harness_rates = failure_rates_config["harness_failure_rates"]
    model_rates = failure_rates_config["model_failure_rates"]

    real = df.copy()
    real["is_synthetic_retry"] = False
    real["error_type"] = None

    _require_mapped(real, harness_rates, model_rates)

    harness_rate = real["harness"].map(harness_rates)
    model_rate = real["model"].map(model_rates)

    # The dataset card only publishes MARGINAL failure rates per harness and
    # per model separately -- it does not publish a joint harness+model rate
    # (e.g. "claude_code running DeepSeek-V3.2").
    #
    # A first version averaged the two marginals: (harness_rate + model_rate) / 2.
    # That was a confirmed structural bug, not just a labeling issue: harness
    # rates and model rates live on very different scales (claude_code is
    # 0.2928; typical model rates are 0.02-0.07), so a plain average always
    # regresses the achieved rate toward the midpoint of those two scales,
    # for every sample size, no matter how much data you throw at it.
    #
    # Fix: use harness_rate as the anchor (the primary, operationally
    # meaningful signal) and scale it by the model's failure rate *relative
    # to the average model* -- a multiplier, not a second absolute rate on a
    # different scale. A model exactly at the average model failure rate
    # leaves the harness rate unchanged (multiplier = 1); a worse-than-average
    # model pushes it up, a better-than-average model pulls it down.
    # The multiplier itself also needs a bound. Without one, an outlier model
    # far above the mean (claude-opus-4-5 at 0.2294 vs. a 0.0762 mean -> a
    # ~3.0x multiplier) combined with even a moderate harness rate can push
    # injection_probability to implausible near-certainty for every call that
    # model makes, regardless of harness -- e.g. claude_code (0.2928) x 3.0x
    # -> ~0.88, meaning ~88% of that model's calls would get a synthetic
    # failure, far beyond anything in the documented statistics. Capping only
    # the final probability doesn't fix this: it just clips the symptom while
    # leaving every below-cap combination for that model still inflated. So
    # model_relative_risk is clipped to a bounded range *before* it's used as
    # a multiplier, keeping any single model's influence on the final rate
    # within a realistic range; the final-probability cap stays as a backstop
    # for the harness-rate x bounded-risk combination, not the primary fix.
    mean_model_rate = sum(model_rates.values()) / len(model_rates)
    model_relative_risk = (model_rate / mean_model_rate).clip(*RELATIVE_RISK_BOUNDS)

    injection_probability = (harness_rate * model_relative_risk).clip(
        upper=INJECTION_PROBABILITY_CAP
    )

    draws = rng.random(len(real))
    selected_mask = draws < injection_probability.to_numpy()
    selected_idx = real.index[selected_mask]
    n_selected = len(selected_idx)

    start_dt = pd.to_datetime(real["start_time"])
    end_dt = pd.to_datetime(real["end_time"])

    # Duration is generated first, and the pre-offset gap is derived from it
    # (duration + a positive buffer) rather than drawn independently. Retries
    # are serialized -- attempt 0 must fully finish before attempt 1 starts --
    # so an independently-drawn gap could be shorter than the duration it's
    # supposed to contain, letting the synthetic attempt's end_time land
    # after the real attempt's start_time. Deriving the gap this way makes
    # that ordering true by construction: gap = duration + buffer, so
    # synth_end = real_start - gap + duration = real_start - buffer, which is
    # always strictly before real_start since buffer > 0.
    synth_duration = rng.uniform(1, 5, size=n_selected)
    buffer = rng.uniform(1, 10, size=n_selected)
    pre_offset_gap = synth_duration + buffer

    sel_start_dt = start_dt.loc[selected_idx]
    synth_start_dt = sel_start_dt - pd.to_timedelta(pre_offset_gap, unit="s")
    synth_end_dt = synth_start_dt + pd.to_timedelta(synth_duration, unit="s")

    synthetic = real.loc[selected_idx].copy()
    synthetic["attempt_id"] = synthetic["attempt_id"].str.replace(
        "_attempt_1", "_attempt_0", regex=False
    )
    synthetic["status_code"] = SYNTHETIC_STATUS_CODE
    synthetic["error_type"] = rng.choice(ERROR_TYPES, size=n_selected)
    synthetic["output_tokens"] = 0
    synthetic["start_time"] = [t.isoformat(timespec="microseconds") for t in synth_start_dt]
    synthetic["end_time"] = [t.isoformat(timespec="microseconds") for t in synth_end_dt]
    synthetic["is_synthetic_retry"] = True

    # Real call was delayed by the time wasted on the synthetic failed attempt.
    new_real_start_dt = sel_start_dt + pd.to_timedelta(synth_duration, unit="s")
    new_real_end_dt = end_dt.loc[selected_idx] + pd.to_timedelta(synth_duration, unit="s")
    real.loc[selected_idx, "start_time"] = [
        t.isoformat(timespec="microseconds") for t in new_real_start_dt
    ]
    real.loc[selected_idx, "end_time"] = [
        t.isoformat(timespec="microseconds") for t in new_real_end_dt
    ]

    combined = pd.concat([real, synthetic], ignore_index=True)
    combined = combined.sort_values(["task_id", "start_time"]).reset_index(drop=True)

    _print_report(
        real, selected_mask, injection_probability, harness_rates, model_rates, mean_model_rate
    )

    return combined


def _print_report(real, selected_mask, injection_probability, harness_rates, model_rates, mean_model_rate):
    total_real = len(real)
    total_synthetic = int(selected_mask.sum())
    overall_rate = total_synthetic / total_real if total_real else 0.0

    print(f"Total real rows in: {total_real}")
    print(f"Total synthetic rows injected: {total_synthetic}")
    print(f"Overall injection rate: {overall_rate:.4f}")
    print(f"mean_model_rate (average across model_failure_rates): {mean_model_rate:.4f}")

    # "target" here is no longer a single marginal rate. It's the mean of
    # each row's own harness_rate * model_relative_risk within the group --
    # i.e. what the achieved rate should converge to given this group's
    # actual harness/model mix. That's the correct expected value to compare
    # against post-fix; the raw marginal harness_rate/model_rate alone is
    # only a reference point now, not the target.
    print("\nPer-harness: actual vs. expected injection rate (harness_rate * mix of model_relative_risk)")
    for harness, harness_rate_ref in harness_rates.items():
        mask = (real["harness"] == harness).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        actual = float(selected_mask[mask].mean())
        expected = float(injection_probability.to_numpy()[mask].mean())
        drift = actual - expected
        flag = " <-- DRIFT > 5pp, likely small-sample variance" if abs(drift) > DRIFT_WARNING_THRESHOLD else ""
        print(
            f"  {harness}: n={n} actual={actual:.4f} expected={expected:.4f} "
            f"(raw harness_rate={harness_rate_ref:.4f}) drift={drift:+.4f}{flag}"
        )

    print("\nPer-model: actual vs. expected injection rate (harness_rate * model_relative_risk)")
    for model, model_rate_ref in model_rates.items():
        mask = (real["model"] == model).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        raw_relative_risk = model_rate_ref / mean_model_rate
        clipped_relative_risk = min(max(raw_relative_risk, RELATIVE_RISK_BOUNDS[0]), RELATIVE_RISK_BOUNDS[1])
        actual = float(selected_mask[mask].mean())
        expected = float(injection_probability.to_numpy()[mask].mean())
        drift = actual - expected
        flag = " <-- DRIFT > 5pp, likely small-sample variance" if abs(drift) > DRIFT_WARNING_THRESHOLD else ""
        clip_note = " [CLIPPED]" if clipped_relative_risk != raw_relative_risk else ""
        print(
            f"  {model}: n={n} relative_risk={clipped_relative_risk:.4f}{clip_note} "
            f"(raw={raw_relative_risk:.4f}) actual={actual:.4f} "
            f"expected={expected:.4f} (raw model_rate={model_rate_ref:.4f}) drift={drift:+.4f}{flag}"
        )


if __name__ == "__main__":
    config = _load_config()
    # shard_0004 is 100% claude_code (harness_rate=0.2928) and includes 2935
    # claude-opus-4-5 rows -- the exact high-harness-rate x outlier-model
    # combination the relative_risk clipping is meant to guard against.
    shard_path = os.path.join(PROCESSED_DIR, "shard_0004_calls.parquet")
    df = pd.read_parquet(shard_path)
    result = inject_synthetic_retries(df, config, random_seed=42)
