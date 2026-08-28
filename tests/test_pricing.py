import pandas as pd
import pytest

from pricing import _require_priced, calculate_cost


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
