import gc
import glob
import os

import pandas as pd
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "pricing.yaml")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

REQUIRED_RATE_KEYS = ("input_token_price_per_million", "output_token_price_per_million")


def _load_pricing_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def calculate_cost(input_tokens, output_tokens, model, pricing_config):
    """
    Given token counts and a model name, look up that model's pricing and
    return the calculated cost in USD.

    Fails loud on an unmapped model -- raises ValueError naming it -- rather
    than silently defaulting to $0 or an average price, same pattern as the
    unmapped-model fix in inject_retries.py's _require_mapped(). A missing
    price should be visible, not quietly absorbed into a wrong total. Same
    treatment for a model that IS present but has a malformed entry (missing
    one of the two required rate keys, e.g. a config typo) -- raises a
    clear, named ValueError instead of a bare KeyError.

    input_tokens/output_tokens may be None or NaN (a synthetic failed row's
    output_tokens is legitimately 0; some real rows have NaN token counts
    for span types that don't report usage) -- treated as 0 tokens, which
    just produces a proportionally lower cost, not an error.
    """
    model_pricing = pricing_config["model_pricing"]
    if model not in model_pricing:
        raise ValueError(f"pricing config is missing a rate for model: {model!r}")

    rates = model_pricing[model]
    missing_keys = [key for key in REQUIRED_RATE_KEYS if key not in rates]
    if missing_keys:
        raise ValueError(
            f"pricing config entry for model {model!r} is missing required key(s): {missing_keys}"
        )

    input_tokens = 0 if pd.isna(input_tokens) else input_tokens
    output_tokens = 0 if pd.isna(output_tokens) else output_tokens

    input_cost = (input_tokens / 1_000_000) * rates["input_token_price_per_million"]
    output_cost = (output_tokens / 1_000_000) * rates["output_token_price_per_million"]
    return input_cost + output_cost


def _require_priced(models, pricing_config):
    """Fail fast, before any per-row computation, if a model in the data
    has no price in pricing_config, or has a malformed entry (missing one
    of the two required rate keys) -- same upfront-check shape as
    inject_retries.py's _require_mapped(), so a bad price is caught
    immediately rather than mid-.apply() partway through a shard.
    calculate_cost() also checks for malformed entries on its own (so a
    direct call still fails loud), but only this upfront pass catches it
    before ANY row has been processed -- calculate_cost()'s check alone
    would let earlier rows in the same .apply() call succeed first.

    Checks for null models first, before building the unmapped-set: a
    mixed set of NaN and genuinely-unmapped string models would otherwise
    reach sorted(unmapped) below, which raises a confusing TypeError
    ('<' not supported between instances of 'float' and 'str') instead of
    a clear ValueError -- the exact edge case inject_retries.py's
    _require_mapped() already guards against with this same ordering."""
    if models.isna().any():
        raise ValueError("Found null model values with no known price.")

    model_pricing = pricing_config["model_pricing"]
    unique_models = set(models.unique())

    unmapped = unique_models - set(model_pricing.keys())
    if unmapped:
        raise ValueError(
            f"model_pricing in config/pricing.yaml is missing a rate for: {sorted(unmapped)}"
        )

    malformed = {
        model: [key for key in REQUIRED_RATE_KEYS if key not in model_pricing[model]]
        for model in sorted(unique_models)
    }
    malformed = {model: keys for model, keys in malformed.items() if keys}
    if malformed:
        raise ValueError(
            f"model_pricing in config/pricing.yaml has entries missing required key(s): {malformed}"
        )


def add_execution_cost(df, pricing_config):
    """Apply calculate_cost() to every row (real and synthetic alike -- a
    synthetic failed row's cost is computed exactly the same way as any
    other row's, no special-casing), returning a copy with a new
    execution_cost_usd column."""
    _require_priced(df["model"], pricing_config)

    result = df.copy()
    result["execution_cost_usd"] = result.apply(
        lambda row: calculate_cost(row["input_tokens"], row["output_tokens"], row["model"], pricing_config),
        axis=1,
    )
    return result


def print_cost_summary(calls):
    """
    Print the full-dataset cost summary -- Ledgr's core metric in action:
    total cost, cost attributable to synthetic (wasted) rows and what share
    of the total that is, and cost broken down by model.

    `calls` needs only model/is_synthetic_retry/execution_cost_usd columns,
    so the same function works on one shard's priced output or the
    lightweight slices accumulated across all 9 shards in
    add_cost_to_all_shards().
    """
    total_cost = float(calls["execution_cost_usd"].sum())
    synthetic_cost = float(calls.loc[calls["is_synthetic_retry"], "execution_cost_usd"].sum())
    synthetic_pct = (synthetic_cost / total_cost * 100) if total_cost else 0.0

    print(f"Total cost across full dataset: ${total_cost:,.2f}")
    print(
        f"Cost attributable to synthetic (wasted) rows: ${synthetic_cost:,.2f} "
        f"({synthetic_pct:.2f}% of total)"
    )

    print("\nCost by model:")
    by_model = calls.groupby("model")["execution_cost_usd"].sum().sort_values(ascending=False)
    for model, cost in by_model.items():
        print(f"  {model}: ${cost:,.2f}")


def add_cost_to_all_shards():
    """
    Apply calculate_cost() to every row of all 9
    data/processed/shard_*_augmented.parquet files, one shard at a time
    (same shard-by-shard memory discipline as process_all_shards() and
    inject_all_shards()), writing each shard's result to
    data/processed/shard_<N>_final.parquet -- the complete, final enriched
    dataset ready for upload.

    Prints a cost summary aggregated across all 9 shards.
    """
    pricing_config = _load_pricing_config()
    shard_paths = sorted(glob.glob(os.path.join(PROCESSED_DIR, "shard_*_augmented.parquet")))

    # Upfront validation across ALL shards, before any output is written.
    # add_execution_cost() already validates per-shard via _require_priced(),
    # but that's too late for atomicity: a bad/malformed model on shard 8
    # would only surface after shards 1-7 were already fully processed and
    # written, leaving a silently incomplete data/processed/ with no signal
    # the run didn't finish. Only the model column is loaded here (not the
    # full shard) to keep this fast.
    coverage_models = pd.concat(
        [pd.read_parquet(p, columns=["model"])["model"] for p in shard_paths],
        ignore_index=True,
    )
    _require_priced(coverage_models, pricing_config)

    cost_slices = []
    output_paths = []

    for shard_path in shard_paths:
        df = pd.read_parquet(shard_path)
        priced = add_execution_cost(df, pricing_config)

        shard_num = os.path.basename(shard_path).replace("_augmented.parquet", "")
        out_path = os.path.join(PROCESSED_DIR, f"{shard_num}_final.parquet")
        priced.to_parquet(out_path, index=False)
        output_paths.append(out_path)

        cost_slices.append(priced[["model", "is_synthetic_retry", "execution_cost_usd"]].copy())

        print(
            f"  {os.path.basename(shard_path)}: {len(priced)} rows -> "
            f"${priced['execution_cost_usd'].sum():,.2f}"
        )

        del df, priced
        gc.collect()

    all_costs = pd.concat(cost_slices, ignore_index=True)
    del cost_slices
    gc.collect()

    print("\n=== Full-dataset cost summary (all 9 shards combined) ===\n")
    print_cost_summary(all_costs)

    all_exist = all(os.path.exists(p) for p in output_paths)
    print(f"\nAll {len(output_paths)} final output files written: {all_exist}")
    for p in output_paths:
        print(f"  {p}")

    return output_paths


if __name__ == "__main__":
    add_cost_to_all_shards()
