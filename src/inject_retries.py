import gc
import glob
import os
import zlib

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


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _extract_calibration(failure_rates_config):
    """
    Pull the tunable injection-calibration parameters out of
    config/failure_rates.yaml (injection_calibration:) rather than reading
    them from hardcoded Python constants. These aren't documented dataset
    statistics like the failure rates -- they're calibration choices -- but
    keeping them in the same config file means every tunable lives in one
    place instead of split between YAML and Python.
    """
    calibration = failure_rates_config["injection_calibration"]
    return {
        "relative_risk_bounds": tuple(calibration["relative_risk_bounds"]),
        "injection_probability_cap": calibration["injection_probability_cap"],
        "drift_warning_threshold": calibration["drift_warning_threshold"],
        "retry_duration_bounds": tuple(calibration["retry_duration_seconds"]),
        "retry_buffer_bounds": tuple(calibration["retry_buffer_seconds"]),
    }


def _stable_uniform(keys, seed, purpose):
    """
    Deterministic per-row uniform draw in [0, 1), keyed by (seed, purpose,
    key) rather than by position in an array.

    The previous version used a single np.random.default_rng(seed) stream
    consumed in row order (rng.random(len(real)), then rng.uniform(...) for
    timing, then rng.choice(...) for error_type). That only reproduces the
    same result if row order never changes -- a future pandas version, a
    reordered json_normalize, or splitting shard processing across parallel
    workers would all silently produce different injected rows under "the
    same seed". Hashing each row's own call_id instead makes the draw a
    pure function of that row's identity: reordering, re-running on a
    subset, or processing shards in a different sequence all leave each
    row's own draw unchanged.

    Not cryptographic -- zlib.crc32 is fast and deterministic across
    processes/platforms, which is all this needs (unlike Python's built-in
    hash(), which is randomized per-process for strings unless
    PYTHONHASHSEED is fixed). `purpose` keys separate draws (selection,
    duration, buffer, error_type) off the same call_id so they don't move
    in lockstep.
    """
    def draw(key):
        digest = zlib.crc32(f"{seed}:{purpose}:{key}".encode())
        return digest / 2**32

    return keys.map(draw)


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


def _compute_injection_probability(harness_series, model_series, harness_rates, model_rates, mean_model_rate, calibration):
    """
    Row-level injection probability: harness_rate as the anchor (the primary,
    operationally meaningful signal), scaled by the model's failure rate
    *relative to the average model* -- a multiplier, not a second absolute
    rate on a different scale.

    The dataset card only publishes MARGINAL failure rates per harness and
    per model separately -- it does not publish a joint harness+model rate
    (e.g. "claude_code running DeepSeek-V3.2"). A first version averaged the
    two marginals: (harness_rate + model_rate) / 2. That was a confirmed
    structural bug, not just a labeling issue: harness rates and model rates
    live on very different scales (claude_code is 0.2928; typical model rates
    are 0.02-0.07), so a plain average always regresses the achieved rate
    toward the midpoint of those two scales, for every sample size, no matter
    how much data you throw at it.

    The multiplier itself also needs a bound. Without one, an outlier model
    far above the mean (claude-opus-4-5 at 0.2294 vs. a 0.0762 mean -> a
    ~3.0x multiplier) combined with even a moderate harness rate can push
    injection_probability to implausible near-certainty for every call that
    model makes, regardless of harness -- e.g. claude_code (0.2928) x 3.0x
    -> ~0.88, meaning ~88% of that model's calls would get a synthetic
    failure, far beyond anything in the documented statistics. Capping only
    the final probability doesn't fix this: it just clips the symptom while
    leaving every below-cap combination for that model still inflated. So
    model_relative_risk is clipped to a bounded range *before* it's used as
    a multiplier, keeping any single model's influence on the final rate
    within a realistic range; the final-probability cap stays as a backstop
    for the harness-rate x bounded-risk combination, not the primary fix.
    """
    harness_rate = harness_series.map(harness_rates)
    model_rate = model_series.map(model_rates)
    model_relative_risk = (model_rate / mean_model_rate).clip(*calibration["relative_risk_bounds"])
    return (harness_rate * model_relative_risk).clip(upper=calibration["injection_probability_cap"])


def inject_synthetic_retries(df, failure_rates_config, random_seed=42, verbose=True):
    """
    Takes the flattened call-level DataFrame and injects synthetic failed-retry
    spans, calibrated jointly on harness AND model, not just harness alone.
    """
    harness_rates = failure_rates_config["harness_failure_rates"]
    model_rates = failure_rates_config["model_failure_rates"]
    mean_model_rate = sum(model_rates.values()) / len(model_rates)
    calibration = _extract_calibration(failure_rates_config)

    real = df.copy()
    real["is_synthetic_retry"] = False
    real["error_type"] = None

    _require_mapped(real, harness_rates, model_rates)

    injection_probability = _compute_injection_probability(
        real["harness"], real["model"], harness_rates, model_rates, mean_model_rate, calibration
    )

    draws = _stable_uniform(real["call_id"], random_seed, "select").to_numpy()
    selected_mask = draws < injection_probability.to_numpy()
    selected_idx = real.index[selected_mask]
    n_selected = len(selected_idx)

    # format="ISO8601" (not the default single-format fast path): a handful
    # of real timestamps land on exactly zero microseconds and are recorded
    # without a fractional-seconds suffix (e.g. "...T16:09:33+00:00" instead
    # of "...T16:09:33.000000+00:00"), which pandas' format-inference can't
    # parse in the same pass as the microsecond-bearing majority.
    start_dt = pd.to_datetime(real["start_time"], format="ISO8601")
    end_dt = pd.to_datetime(real["end_time"], format="ISO8601")

    selected_call_ids = real.loc[selected_idx, "call_id"]

    # Duration is generated first, and the pre-offset gap is derived from it
    # (duration + a positive buffer) rather than drawn independently. Retries
    # are serialized -- attempt 0 must fully finish before attempt 1 starts --
    # so an independently-drawn gap could be shorter than the duration it's
    # supposed to contain, letting the synthetic attempt's end_time land
    # after the real attempt's start_time. Deriving the gap this way makes
    # that ordering true by construction: gap = duration + buffer, so
    # synth_end = real_start - gap + duration = real_start - buffer, which is
    # always strictly before real_start since buffer > 0.
    duration_lo, duration_hi = calibration["retry_duration_bounds"]
    buffer_lo, buffer_hi = calibration["retry_buffer_bounds"]
    synth_duration = duration_lo + _stable_uniform(selected_call_ids, random_seed, "duration").to_numpy() * (
        duration_hi - duration_lo
    )
    buffer = buffer_lo + _stable_uniform(selected_call_ids, random_seed, "buffer").to_numpy() * (
        buffer_hi - buffer_lo
    )
    pre_offset_gap = synth_duration + buffer

    sel_start_dt = start_dt.loc[selected_idx]
    synth_start_dt = sel_start_dt - pd.to_timedelta(pre_offset_gap, unit="s")
    synth_end_dt = synth_start_dt + pd.to_timedelta(synth_duration, unit="s")

    synthetic = real.loc[selected_idx].copy()
    synthetic["attempt_id"] = synthetic["attempt_id"].str.replace(
        "_attempt_1", "_attempt_0", regex=False
    )
    synthetic["status_code"] = SYNTHETIC_STATUS_CODE
    error_draws = _stable_uniform(selected_call_ids, random_seed, "error_type").to_numpy()
    error_index = np.minimum((error_draws * len(ERROR_TYPES)).astype(int), len(ERROR_TYPES) - 1)
    synthetic["error_type"] = np.array(ERROR_TYPES)[error_index]
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

    if verbose:
        _print_report(
            combined[["harness", "model", "is_synthetic_retry"]],
            harness_rates,
            model_rates,
            mean_model_rate,
            calibration,
        )

    return combined


def _print_report(calls, harness_rates, model_rates, mean_model_rate, calibration):
    """
    Print total/per-harness/per-model actual-vs-expected injection rates.

    `calls` only needs harness/model/is_synthetic_retry columns -- both real
    and synthetic rows carry them (synthetic rows copy their real
    predecessor's), so "actual" is fully recoverable by counting, with no
    need to also thread a selected_mask array through. This is what makes the
    same function work for a single shard's combined output AND for the
    lightweight (harness, model, is_synthetic_retry) slices accumulated
    across all 9 shards in inject_all_shards -- same computation either way,
    just a bigger `calls`.
    """
    real_mask = ~calls["is_synthetic_retry"]
    total_real = int(real_mask.sum())
    total_synthetic = int((~real_mask).sum())
    overall_rate = total_synthetic / total_real if total_real else 0.0

    # A small DataFrame of just the real rows, with each row's own injection
    # probability attached as a column -- avoids re-aligning a separately
    # indexed Series against boolean masks below; just filter this and average.
    real_calls = calls.loc[real_mask, ["harness", "model"]].copy()
    real_calls["injection_probability"] = _compute_injection_probability(
        real_calls["harness"], real_calls["model"], harness_rates, model_rates, mean_model_rate, calibration
    )

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
        group = real_calls[real_calls["harness"] == harness]
        n = len(group)
        if n == 0:
            continue
        n_synth = int((~real_mask & (calls["harness"] == harness)).sum())
        actual = n_synth / n
        expected = float(group["injection_probability"].mean())
        drift = actual - expected
        flag = " <-- DRIFT > 5pp, likely small-sample variance" if abs(drift) > calibration["drift_warning_threshold"] else ""
        print(
            f"  {harness}: n={n} actual={actual:.4f} expected={expected:.4f} "
            f"(raw harness_rate={harness_rate_ref:.4f}) drift={drift:+.4f}{flag}"
        )

    print("\nPer-model: actual vs. expected injection rate (harness_rate * model_relative_risk)")
    for model, model_rate_ref in model_rates.items():
        group = real_calls[real_calls["model"] == model]
        n = len(group)
        if n == 0:
            continue
        n_synth = int((~real_mask & (calls["model"] == model)).sum())
        raw_relative_risk = model_rate_ref / mean_model_rate
        bounds = calibration["relative_risk_bounds"]
        clipped_relative_risk = min(max(raw_relative_risk, bounds[0]), bounds[1])
        actual = n_synth / n
        expected = float(group["injection_probability"].mean())
        drift = actual - expected
        flag = " <-- DRIFT > 5pp, likely small-sample variance" if abs(drift) > calibration["drift_warning_threshold"] else ""
        clip_note = " [CLIPPED]" if clipped_relative_risk != raw_relative_risk else ""
        print(
            f"  {model}: n={n} relative_risk={clipped_relative_risk:.4f}{clip_note} "
            f"(raw={raw_relative_risk:.4f}) actual={actual:.4f} "
            f"expected={expected:.4f} (raw model_rate={model_rate_ref:.4f}) drift={drift:+.4f}{flag}"
        )


def inject_all_shards():
    """
    Apply inject_synthetic_retries() to all 9 data/processed/shard_*_calls.parquet
    files, one shard at a time (same shard-by-shard memory discipline as
    process_all_shards() in explode_spans.py -- load, process, write, drop,
    move on, never hold more than one shard's data in memory at once). Each
    shard's augmented output (real + synthetic rows combined) is written to
    data/processed/shard_<N>_augmented.parquet.

    Prints a final summary aggregated across all 9 shards: total real rows,
    total synthetic rows, overall injection rate, per-harness and per-model
    actual-vs-expected rates computed over the full dataset (not one shard),
    and a confirmation that zero timing inversions occurred anywhere.
    """
    config = _load_config()
    harness_rates = config["harness_failure_rates"]
    model_rates = config["model_failure_rates"]
    mean_model_rate = sum(model_rates.values()) / len(model_rates)
    calibration = _extract_calibration(config)

    shard_paths = sorted(glob.glob(os.path.join(PROCESSED_DIR, "shard_*_calls.parquet")))

    # Lightweight per-shard accumulator for the final aggregate report --
    # only harness/model/is_synthetic_retry, not the full augmented rows, so
    # this stays cheap even across all 9 shards combined.
    report_slices = []
    total_inversions = 0
    output_paths = []

    for shard_path in shard_paths:
        df = pd.read_parquet(shard_path)
        # Original (pre-injection) start times, keyed by call_id, to check
        # timing inversions against -- inject_synthetic_retries shifts the
        # real row's start_time, so this has to be captured before that.
        original_start = df.set_index("call_id")["start_time"]

        combined = inject_synthetic_retries(df, config, random_seed=42, verbose=False)

        synth = combined[combined["is_synthetic_retry"]]
        synth_end_dt = pd.to_datetime(synth["end_time"], format="ISO8601")
        real_orig_start_dt = pd.to_datetime(synth["call_id"].map(original_start), format="ISO8601")
        total_inversions += int((synth_end_dt >= real_orig_start_dt).sum())

        report_slices.append(combined[["harness", "model", "is_synthetic_retry"]].copy())

        shard_num = os.path.basename(shard_path).replace("_calls.parquet", "")
        out_path = os.path.join(PROCESSED_DIR, f"{shard_num}_augmented.parquet")
        combined.to_parquet(out_path, index=False)
        output_paths.append(out_path)

        print(f"  {os.path.basename(shard_path)}: {len(df)} real -> {len(combined)} total ({len(synth)} synthetic)")

        del df, combined, synth
        gc.collect()

    all_calls = pd.concat(report_slices, ignore_index=True)
    del report_slices
    gc.collect()

    print("\n=== Full-dataset summary (all 9 shards combined) ===\n")
    _print_report(all_calls, harness_rates, model_rates, mean_model_rate, calibration)

    print(f"\nZero timing inversions across full dataset: {total_inversions == 0} (inversions={total_inversions})")

    all_exist = all(os.path.exists(p) for p in output_paths)
    print(f"\nAll {len(output_paths)} augmented shard files written: {all_exist}")
    for p in output_paths:
        print(f"  {p}")

    return output_paths


if __name__ == "__main__":
    inject_all_shards()
