import glob
import os

import pandas as pd
import pytest

from explode_spans import process_shard
from load_raw import get_shard_paths, load_shard

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))

EXPECTED_COLUMNS = {
    "task_id", "run_id", "trace_id", "call_id", "attempt_id", "span_name",
    "start_time", "end_time", "status_code", "status_message", "model",
    "input_tokens", "output_tokens", "provider", "input_message_length",
    "output_message_length", "has_tool_definitions", "harness", "benchmark",
    "success", "agent_cost", "execution_time",
}


def _smallest_raw_shard():
    """The smallest shard (0000.parquet, 150 sessions) so this stays fast.
    Skips cleanly if data/raw/ isn't present -- it's gitignored (large
    downloaded data, not source), so a fresh checkout without the dataset
    downloaded shouldn't fail the whole suite."""
    paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.parquet")))
    if not paths:
        pytest.skip("data/raw/*.parquet not found -- download the dataset first")
    return paths[0]


def test_get_shard_paths_finds_raw_shards():
    paths = get_shard_paths()
    if not paths:
        pytest.skip("data/raw/*.parquet not found -- download the dataset first")
    assert paths == sorted(paths)
    assert all(p.endswith(".parquet") for p in paths)


def test_load_shard_has_expected_raw_columns():
    shard_path = _smallest_raw_shard()
    raw = load_shard(shard_path)
    assert "spans" in raw.columns
    assert "session_id" in raw.columns
    assert len(raw) > 0


def test_process_shard_row_and_column_count():
    """The exact class of bug this module has had before (a silent row
    drop, a schema mismatch) would show up here: an independent Python-level
    count of spans in the raw file, cross-checked against process_shard()'s
    vectorized explode+json_normalize row count."""
    shard_path = _smallest_raw_shard()

    raw = load_shard(shard_path)
    independent_span_count = sum(len(spans) for spans in raw["spans"] if spans is not None)

    result = process_shard(shard_path)

    assert set(result.columns) == EXPECTED_COLUMNS
    assert len(result) > 0
    assert len(result) == independent_span_count


def test_process_shard_known_values():
    """Locks in the vectorized explode/json_normalize logic against a
    known-correct sample: the first row of the smallest raw shard, verified
    by hand early in this project and re-verified against the current
    implementation."""
    shard_path = _smallest_raw_shard()
    if os.path.basename(shard_path) != "0000.parquet":
        pytest.skip("golden values below are specific to 0000.parquet")

    result = process_shard(shard_path)
    first = result.iloc[0]

    assert first["call_id"] == "29f911a143e3283a"
    assert first["trace_id"] == "4caf14f2a545ad8bbabe39a4264eb575"
    assert first["model"] == "DeepSeek-V3.2"
    assert first["input_tokens"] == 112814.0
    assert first["output_tokens"] == 234.0
    assert first["status_code"] == 1
    assert first["harness"] == "claude_code"
    assert first["benchmark"] == "appworld"
    assert first["attempt_id"] == "29f911a143e3283a_attempt_1"
