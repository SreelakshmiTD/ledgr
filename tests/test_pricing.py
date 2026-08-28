import pandas as pd
import pytest

from pricing import _require_priced, add_execution_cost, calculate_cost


def test_calculate_cost_basic(pricing_config):
    cost = calculate_cost(1_000_000, 0, "claude-opus-4-5", pricing_config)
    expected = pricing_config["model_pricing"]["claude-opus-4-5"]["input_token_price_per_million"]
    assert cost == pytest.approx(expected)


def test_unpriced_model_raises(pricing_config):
    with pytest.raises(ValueError, match="totally-unknown-model"):
        calculate_cost(1000, 100, "totally-unknown-model", pricing_config)


def test_nan_model_raises_valueerror_not_typeerror(pricing_config):
    """Reproduces the exact bug found earlier: a NaN model coexisting with a
    genuinely-unmapped model crashed sorted() with an unclear TypeError
    ('<' not supported between instances of 'float' and 'str') instead of
    the intended clear ValueError. Asserts the fix's specific NaN-first
    message fires, not just that some ValueError happens to occur."""
    models = pd.Series(["claude-opus-4-5", float("nan"), "totally-unknown-model"])
    with pytest.raises(ValueError, match="null model"):
        _require_priced(models, pricing_config)


def test_zero_tokens_handled(pricing_config):
    cost_with_output = calculate_cost(1000, 500, "claude-opus-4-5", pricing_config)
    cost_zero_output = calculate_cost(1000, 0, "claude-opus-4-5", pricing_config)

    assert cost_zero_output < cost_with_output

    rates = pricing_config["model_pricing"]["claude-opus-4-5"]
    expected_zero = (1000 / 1_000_000) * rates["input_token_price_per_million"]
    assert cost_zero_output == pytest.approx(expected_zero)

    # None is also valid (a real synthetic-row scenario), not just 0
    cost_none_output = calculate_cost(1000, None, "claude-opus-4-5", pricing_config)
    assert cost_none_output == pytest.approx(expected_zero)


def test_malformed_pricing_entry_raises_valueerror_not_keyerror():
    """Reproduces the bug found during the previous review: a model present
    in model_pricing but missing one of its two required rate keys raised a
    bare KeyError ('output_token_price_per_million') instead of the clear,
    named ValueError this codebase uses everywhere else for config
    problems (_require_mapped, _require_priced)."""
    bad_config = {"model_pricing": {"broken-model": {"input_token_price_per_million": 1.0}}}
    with pytest.raises(ValueError, match="broken-model"):
        calculate_cost(1000, 500, "broken-model", bad_config)


def test_add_execution_cost_adds_correct_column(pricing_config, calls_factory):
    """Exercises add_execution_cost() itself -- the function actually used
    by the pipeline -- not just the lower-level calculate_cost() it wraps."""
    df = calls_factory(5, "claude_code", "claude-opus-4-5")
    priced = add_execution_cost(df, pricing_config)

    assert "execution_cost_usd" in priced.columns
    assert len(priced) == len(df)
    for _, row in priced.iterrows():
        expected = calculate_cost(row["input_tokens"], row["output_tokens"], row["model"], pricing_config)
        assert row["execution_cost_usd"] == pytest.approx(expected)


def test_add_cost_to_all_shards_writes_correct_output(tmp_path, monkeypatch, calls_factory):
    """Exercises the actual pipeline entry point end-to-end (file read ->
    cost calculation -> file write), not just add_execution_cost() in
    memory -- catches bugs in the glob pattern / output path / naming that
    reading pre-existing data/processed/ files would never catch."""
    import pricing

    df = calls_factory(5, "claude_code", "claude-opus-4-5")
    df["is_synthetic_retry"] = False
    df["error_type"] = None
    input_path = tmp_path / "shard_0000_augmented.parquet"
    df.to_parquet(input_path, index=False)

    monkeypatch.setattr(pricing, "PROCESSED_DIR", str(tmp_path))

    output_paths = pricing.add_cost_to_all_shards()

    expected_output = tmp_path / "shard_0000_final.parquet"
    assert str(expected_output) in output_paths
    assert expected_output.exists()

    result = pd.read_parquet(expected_output)
    assert set(result.columns) == set(df.columns) | {"execution_cost_usd"}
    assert len(result) == len(df)
    assert not result["execution_cost_usd"].isna().any()
    assert not (result["execution_cost_usd"] < 0).any()


def test_upfront_validation_prevents_partial_output(tmp_path, monkeypatch, pricing_config):
    """Reproduces the exact partial-write scenario found in the prior
    review: 3 shards, 2 with a valid model and 1 with an unmapped model.

    Before the atomicity fix, add_cost_to_all_shards() wrote
    shard_0000_final.parquet and shard_0001_final.parquet -- fully valid,
    normal-looking output -- before crashing on shard 2, leaving a
    silently incomplete data/processed/ with no signal the run didn't
    finish. After the fix, the upfront validation pass (across all
    shards' models, before any shard is processed or any file written)
    raises immediately and writes ZERO output files."""
    import pricing

    known_model = next(iter(pricing_config["model_pricing"].keys()))

    for i, model in enumerate([known_model, known_model, "totally-unmapped-model"]):
        df = pd.DataFrame(
            {
                "model": [model] * 10,
                "input_tokens": [1000] * 10,
                "output_tokens": [500] * 10,
                "is_synthetic_retry": [False] * 10,
            }
        )
        df.to_parquet(tmp_path / f"shard_{i:04d}_augmented.parquet", index=False)

    monkeypatch.setattr(pricing, "PROCESSED_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="totally-unmapped-model"):
        pricing.add_cost_to_all_shards()

    written_final_files = list(tmp_path.glob("*_final.parquet"))
    assert written_final_files == [], f"expected zero output files, found: {written_final_files}"
