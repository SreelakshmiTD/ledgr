# ADR-0002: Bronze CHECK constraint scope

## Context

This ADR documents a decision made in the Databricks/Unity Catalog portion of the
Ledgr project (a separate environment/codebase from this local pandas pipeline).
The local pipeline in this repo produces the enriched dataset that is uploaded to
S3 and ingested into a Databricks Bronze/Silver/Gold Delta Lake architecture as a
downstream stage. The constraint additions and rejection tests described below were
run and verified directly in that Databricks environment by the project owner, not
by Claude Code in this repository.

## Decision

The Databricks Bronze table (`ledgr.bronze.sessions_raw`) has three Delta CHECK
constraints: session_id_not_null, run_id_not_null, and harness_known_value
(restricting harness to the 5 documented values from the Exgentic dataset).
Each was verified by attempting to insert a row that violated it and confirming
a DeltaInvariantViolationException was raised with row count unchanged.

## Why harness_known_value was added

An unrecognized harness value would silently break the Databricks Silver-layer
calibration logic downstream (an analogous fail-loud pattern to this repo's own
_require_mapped() in inject_retries.py and _require_priced() in pricing.py).
Catching this at the earliest ingestion point is a fail-fast posture.

## Why numeric/cost constraints were NOT added at Bronze

Fields like agent_cost are raw passthrough values at Bronze. Constraining them
there would contradict Bronze's purpose of preserving source data as-is. Such
constraints belong on Ledgr's own computed fields at Silver/Gold, not Bronze.
