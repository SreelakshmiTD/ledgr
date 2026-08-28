import glob
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def get_shard_paths():
    """Return the sorted list of parquet shard paths in data/raw/."""
    return sorted(glob.glob(os.path.join(RAW_DIR, "*.parquet")))


def load_shard(shard_path):
    """
    Load a single parquet shard into a DataFrame.

    Loading all 9 shards at once and concatenating them OOM-kills this machine:
    the 'spans' column holds full nested agent trace payloads (some single
    fields run 400K+ characters) that balloon once deserialized. Shards must
    be processed one at a time — see explode_spans.py.
    """
    return pd.read_parquet(shard_path)
