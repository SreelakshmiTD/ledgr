# Data dictionary: `data/processed/shard_*_final.parquet`

25 columns, one row per call attempt (real or synthetic). Types are as
written by `to_parquet()`/read back by pandas; `str` columns are
pyarrow-backed string, not Python `object`.

| Column | Type | Description |
|---|---|---|
| `task_id` | str | The agent session this call belongs to (source dataset's `session_id`). Multiple calls share one `task_id`. |
| `run_id` | str | Groups related sessions/conversations under one run (source dataset's `run_id`). |
| `trace_id` | str | OpenTelemetry trace ID for this span. |
| `call_id` | str | The span's own ID (source `span_id`). Unique per call attempt — a real call and its injected synthetic predecessor share the same `call_id` but differ in `attempt_id`. |
| `attempt_id` | str | `{call_id}_attempt_1` for the real call, `{call_id}_attempt_0` for its synthetic failed predecessor (if one was injected). |
| `span_name` | str | e.g. `"chat openai/azure/DeepSeek-V3.2"` — the span's own name as recorded by the harness's OTel instrumentation. |
| `start_time` | str (ISO 8601, UTC, microsecond precision) | Call start. For a real row that got a synthetic predecessor injected, this is shifted *later* than the original raw value by the synthetic attempt's duration, to represent the real call being delayed by the wasted retry. |
| `end_time` | str (ISO 8601, UTC, microsecond precision) | Call end, same shifting rule as `start_time`. |
| `status_code` | int64 | OpenTelemetry status code: `0`=UNSET, `1`=OK, `2`=ERROR. All injected synthetic rows are `2`. All three values occur in real data. |
| `status_message` | str | OTel status message. Empty string on essentially every real row observed; not populated for synthetic rows either. |
| `model` | str | Model name as recorded by the harness (e.g. `claude-opus-4-5`, `DeepSeek-V3.2`). Matches the keys in `config/failure_rates.yaml` and `config/pricing.yaml`. |
| `input_tokens` | float64 (nullable) | Input token count for this call. NaN on some real spans that don't report usage (observed to correlate with `status_code=2`, i.e. non-OK spans). Copied unchanged onto a synthetic row from its real predecessor — a failed retry is modeled as having sent the same input. |
| `output_tokens` | float64 (nullable) | Output token count. Always exactly `0` on synthetic rows (the call failed before producing output). NaN on the same real-span cases as `input_tokens`. |
| `provider` | str | Who actually served/billed the call, e.g. `azure.ai.openai`, `anthropic`, `gcp.gemini` — the dataset's own `attributes.gen_ai.provider.name`. **Load-bearing for cost accuracy**: 3 of 5 models in this dataset (`DeepSeek-V3.2`, `Kimi-K2.5`, `gpt-5.2-2025-12-11`) are billed via `azure.ai.openai`, not each model-maker's own direct API — `config/pricing.yaml` prices by this field's vendor, not by who trained the model. See `docs/adr/0001-synthetic-retry-injection-calibration.md` for the correction history. |
| `input_message_length` | int64 | Character count of the raw JSON-serialized input messages string (`gen_ai.input.messages`). **Characters, not tokens** — a rough size signal kept instead of the full message content to avoid the memory blowup documented in `docs/adr/0001-synthetic-retry-injection-calibration.md`. |
| `output_message_length` | int64 | Same, for output messages. Always `0` on synthetic rows, alongside `output_tokens` (a failed call produces no output). |
| `has_tool_definitions` | bool | Whether this call's request included tool/function definitions. |
| `harness` | str | Agent framework/harness used (e.g. `claude_code`, `openai_solo`). Matches the keys in `config/failure_rates.yaml`'s `harness_failure_rates`. |
| `benchmark` | str | Benchmark/environment name the session was run against (e.g. `appworld`). |
| `success` | bool | **Session-level** outcome — the same value repeats across every row of a `task_id`, real and synthetic alike. Not a per-call success flag: a synthetic row with `status_code=2` can still show `success=True` if the session ultimately succeeded after the (simulated) retry. Don't filter on this expecting call-level semantics. |
| `agent_cost` | float64 | **Session-level**, repeated per row. The *source dataset's own* self-reported cost estimate in USD. Verified against a real session (145 `DeepSeek-V3.2` calls, 3,999,786 input / 6,818 output tokens): `agent_cost` = 0.8316, which is within 0.68% of that session's cost computed at DeepSeek's *direct* API rate ($0.2088/$0.3096 per M) — not the Azure-marketplace rate ($0.580/$1.68 per M) this project's own `execution_cost_usd` uses for that same model. In other words: **`agent_cost` and `execution_cost_usd` are independently computed and not expected to reconcile**, especially for the 3 Azure-billed models, where they can differ by ~2-3x. `agent_cost` also predates (and never includes) the synthetic retry rows this pipeline adds. |
| `execution_time` | float64 | **Session-level**, repeated per row. The source dataset's own self-reported session duration. **Verified in seconds** (not ms): for the same sample session, `execution_time`=2305.65 vs. the actual wall-clock span coverage (last `end_time` minus first `start_time`) of 1805.03 — same order of magnitude (tens of minutes), with `execution_time` somewhat larger, plausibly because it includes non-LLM time (setup, grading) the spans don't cover. Milliseconds would imply a ~2.3-second task; hours would imply a ~96-day task — both implausible for an agent session with 145 LLM calls. |
| `is_synthetic_retry` | bool | `False` for every real row, `True` for every injected synthetic row. See `docs/adr/0001-synthetic-retry-injection-calibration.md` for calibration, and the `provider`/`output_message_length`/`agent_cost` rows above for specific things that do and don't get overridden on a synthetic row. |
| `error_type` | str (nullable) | `None`/null on every real row. One of `rate_limit`, `timeout`, `malformed_response` on synthetic rows (uniformly random per synthetic row, keyed deterministically to `call_id`). |
| `execution_cost_usd` | float64 | **Per-row** (not session-level, unlike `agent_cost`) — this call's own cost in USD, computed from `input_tokens`/`output_tokens` and `config/pricing.yaml`'s rate for `model`. Computed identically for real and synthetic rows (no special-casing) — a synthetic row's cost is genuinely counted as wasted spend. This is the column `docs/adr/0001-synthetic-retry-injection-calibration.md`'s "pricing accuracy" limitation section applies to: confidence varies per model, see `config/pricing.yaml`'s `pricing_source_note` for each. |

## Two cost columns, two different things

`agent_cost` (session-level, from the source dataset, direct-vendor
pricing, no synthetic rows) and `execution_cost_usd` (per-call, this
project's own pricing, includes synthetic rows) measure related but
genuinely different things. Don't sum `execution_cost_usd` per session and
compare it to `agent_cost` expecting a match — they're not computing the
same quantity, and the gap for Azure-billed models is confirmed to be
large (the verification above found a real session where the two would
differ by roughly 2.8x, entirely explained by direct-vendor vs.
Azure-marketplace pricing, not an error in either field).
