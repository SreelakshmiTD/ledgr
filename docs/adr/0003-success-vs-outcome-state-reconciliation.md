# ADR-0003: success vs outcome_state reconciliation finding

## Context

This ADR documents a finding from the Databricks/Unity Catalog Silver layer
(a separate environment/codebase from this local pandas pipeline, same
context as ADR-0002). The Silver table ledgr.silver.calls_enriched carries
two success-related fields at different grains: `success` (session-level,
inherited from Bronze/the source dataset) and `outcome_state` (call-level,
derived from status_code).

## Finding

A reconciliation check comparing these two fields found a 63.07% mismatch
rate across 10,056 sessions (6,342 sessions where session-level success
and call-level outcome_state do not fully align).

## Interpretation

This is NOT a data quality bug. The two fields answer different questions:
- `success` (session-level): whether the agent's benchmark task was judged
  complete/correct (a semantic, task-completion outcome).
- `outcome_state` (call-level): whether each individual API call executed
  without error (an infrastructure/execution outcome).

A session can have all technically-successful API calls while still
failing its actual task (e.g., an agent successfully executes every tool
call but produces a wrong final answer). Conversely, a session can have
some failed/retried calls while still ultimately completing its task
correctly. The high mismatch rate confirms these are genuinely distinct,
valid signals, not the same fact measured twice.

## Decision

No fix applied, none needed. Both fields are retained as-is, with this
distinction explicitly documented rather than left as unexplained
ambiguity. A lightweight tracking mart (mart_success_outcome_reconciliation)
was added to keep this reconciliation rate visible going forward, so a
dramatic future shift in the rate (which could indicate an actual pipeline
issue) would be noticeable rather than silent.
