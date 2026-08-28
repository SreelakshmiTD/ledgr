import glob
import os

import pandas as pd
import pytest

PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))


def _load_final_dataset():
    paths = sorted(glob.glob(os.path.join(PROCESSED_DIR, "shard_*_final.parquet")))
    if not paths:
        pytest.skip("data/processed/shard_*_final.parquet not found -- run the pipeline first")
    frames = [pd.read_parquet(p, columns=["model", "execution_cost_usd"]) for p in paths]
    return pd.concat(frames, ignore_index=True)


def test_total_cost_reconciles():
    df = _load_final_dataset()
    total = df["execution_cost_usd"].sum()
    by_model_sum = df.groupby("model")["execution_cost_usd"].sum().sum()
    assert total == pytest.approx(by_model_sum, abs=1e-6)


def test_no_negative_or_null_costs():
    df = _load_final_dataset()
    assert not (df["execution_cost_usd"] < 0).any()
    assert not df["execution_cost_usd"].isna().any()
