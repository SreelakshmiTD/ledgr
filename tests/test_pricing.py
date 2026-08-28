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
