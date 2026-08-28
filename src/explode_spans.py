import gc
import os
import platform
import resource
import time

import pandas as pd

from load_raw import get_shard_paths, load_shard

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def _peak_rss_mb():
    """Rough peak RSS in MB. ru_maxrss is bytes on macOS, KB on Linux."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if platform.system() == "Darwin" else peak / 1024


def _col(flat, name):
    """Fetch a column from a json_normalize() result, or an all-NaN stand-in
    if no span in this shard had that key (json_normalize only creates
    columns for keys it actually saw)."""
    if name in flat.columns:
        return flat[name]
    return pd.Series([None] * len(flat), index=flat.index)


def process_shard(shard_path):
    """
    Process ONE parquet shard at a time: explode each session's 'spans' array
    into individual rows, keeping only lightweight fields (never the full
    'attributes' dict, never gen_ai.tool.definitions or full message content).
    Returns a flat DataFrame, one row per span.

    Vectorized: df.explode('spans') expands the array column without a
    Python loop, and pd.json_normalize() flattens each span dict (including
    nested 'status' and 'attributes') into columns in one bulk pass. This
    replaces the previous iterrows() version, which took ~80 minutes for
    241K rows because iterrows() rebuilds a full mixed-type Series for every
    row of the entire DataFrame.
    """
    df = load_shard(shard_path)
    session_count = len(df)

    exploded = df.explode("spans")
    exploded = exploded[exploded["spans"].notna()].reset_index(drop=True)

    del df
    gc.collect()

    flat = pd.json_normalize(exploded["spans"].tolist())

    span_id = _col(flat, "span_id")

    req_model = _col(flat, "attributes.gen_ai.request.model")
    resp_model = _col(flat, "attributes.gen_ai.response.model")
    model = req_model.where(req_model.notna() & (req_model != ""), resp_model)

    input_messages = _col(flat, "attributes.gen_ai.input.messages").fillna("").astype(str)
    output_messages = _col(flat, "attributes.gen_ai.output.messages").fillna("").astype(str)
    tool_definitions = _col(flat, "attributes.gen_ai.tool.definitions")

    result = pd.DataFrame(
        {
            "task_id": exploded["session_id"].to_numpy(),
            "run_id": exploded["run_id"].to_numpy(),
            "trace_id": _col(flat, "trace_id").to_numpy(),
            "call_id": span_id.to_numpy(),
            "attempt_id": (span_id.astype(str) + "_attempt_1").to_numpy(),
            "span_name": _col(flat, "name").to_numpy(),
            "start_time": _col(flat, "start_time").to_numpy(),
            "end_time": _col(flat, "end_time").to_numpy(),
            "status_code": _col(flat, "status.code").to_numpy(),
            "status_message": _col(flat, "status.message").to_numpy(),
            "model": model.to_numpy(),
            "input_tokens": _col(flat, "attributes.gen_ai.usage.input_tokens").to_numpy(),
            "output_tokens": _col(flat, "attributes.gen_ai.usage.output_tokens").to_numpy(),
            "provider": _col(flat, "attributes.gen_ai.provider.name").to_numpy(),
            "input_message_length": input_messages.str.len().to_numpy(),
            "output_message_length": output_messages.str.len().to_numpy(),
            "has_tool_definitions": (tool_definitions.notna() & (tool_definitions.fillna("") != "")).to_numpy(),
            "harness": exploded["harness"].to_numpy(),
            "benchmark": exploded["benchmark"].to_numpy(),
            "success": exploded["success"].to_numpy(),
            "agent_cost": exploded["agent_cost"].to_numpy(),
            "execution_time": exploded["execution_time"].to_numpy(),
        }
    )

    del exploded, flat
    gc.collect()

    print(
        f"  {os.path.basename(shard_path)}: "
        f"{session_count} sessions -> {len(result)} spans "
        f"| peak RSS so far: {_peak_rss_mb():.0f} MB"
    )

    return result


def process_all_shards():
    """
    Process all 9 shards in data/raw/ one at a time, writing each shard's
    exploded output straight to data/processed/shard_<N>_calls.parquet
    instead of accumulating all shards in memory.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    shard_paths = get_shard_paths()

    total_rows = 0
    output_paths = []
    start = time.monotonic()

    for shard_path in shard_paths:
        shard_df = process_shard(shard_path)

        shard_num = os.path.splitext(os.path.basename(shard_path))[0]
        out_path = os.path.join(PROCESSED_DIR, f"shard_{shard_num}_calls.parquet")
        shard_df.to_parquet(out_path, index=False)
        output_paths.append(out_path)

        total_rows += len(shard_df)

        del shard_df
        gc.collect()

    elapsed = time.monotonic() - start
    print(f"\nTotal runtime: {elapsed:.1f}s ({elapsed / 60:.2f} min)")
    print(f"Total exploded rows across all shards: {total_rows}")

    all_exist = all(os.path.exists(p) for p in output_paths)
    print(f"All {len(output_paths)} output files written: {all_exist}")
    for p in output_paths:
        print(f"  {p}")

    return output_paths


if __name__ == "__main__":
    process_all_shards()
