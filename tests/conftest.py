import os
import sys

import pandas as pd
import pytest
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


@pytest.fixture
def failure_rates_config():
    """The real config/failure_rates.yaml -- tests exercise the actual
    production config (real rates, real calibration bounds), not a
    hand-rolled stand-in that could drift from what's actually deployed."""
    path = os.path.join(REPO_ROOT, "config", "failure_rates.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def pricing_config():
    """The real config/pricing.yaml, same rationale as failure_rates_config."""
    path = os.path.join(REPO_ROOT, "config", "pricing.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def _make_calls_df(n, harness, model, start_base="2026-01-01T00:00:00.000000+00:00"):
    """Build a small, schema-correct call-level DataFrame (the same 22
    columns explode_spans.py produces) for testing, with n rows all sharing
    one harness/model combination so injection probability is uniform and
    predictable within a test."""
    start = pd.Timestamp(start_base)
    rows = []
    for i in range(n):
        call_id = f"call_{harness}_{model}_{i:05d}"
        row_start = start + pd.Timedelta(seconds=i * 60)
        row_end = row_start + pd.Timedelta(seconds=30)
        rows.append(
            {
                "task_id": f"task_{i:05d}",
                "run_id": f"run_{i:05d}",
                "trace_id": f"trace_{i:05d}",
                "call_id": call_id,
                "attempt_id": f"{call_id}_attempt_1",
                "span_name": f"chat openai/{model}",
                "start_time": row_start.isoformat(timespec="microseconds"),
                "end_time": row_end.isoformat(timespec="microseconds"),
                "status_code": 1,
                "status_message": "",
                "model": model,
                "input_tokens": 1000.0 + i,
                "output_tokens": 100.0 + i,
                "provider": "test-provider",
                "input_message_length": 500,
                "output_message_length": 50,
                "has_tool_definitions": True,
                "harness": harness,
                "benchmark": "test-benchmark",
                "success": True,
                "agent_cost": 1.0,
                "execution_time": 60.0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def calls_factory():
    """Returns _make_calls_df so tests can build a sample DataFrame with
    whatever row count/harness/model combination they need."""
    return _make_calls_df
