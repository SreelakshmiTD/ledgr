# 1. Calibrating synthetic retry injection to harness and model failure rates

## Status

Accepted (revised)

## Context

`src/inject_retries.py` injects synthetic failed-retry spans into the flattened
call-level data, so downstream analysis reflects realistic retry overhead. The
Exgentic dataset card publishes two separate marginal failure rates in
`config/failure_rates.yaml`:

- `harness_failure_rates` (e.g. `claude_code: 0.2928`)
- `model_failure_rates` (e.g. `DeepSeek-V3.2: 0.0574`, `claude-opus-4-5: 0.2294`)

There is no published *joint* harness+model failure rate (e.g. "claude_code
running DeepSeek-V3.2specifically"). Each real call has both a harness and a
model, so injection has to be calibrated on both dimensions from these two
marginals alone.

## Decision (first attempt, rejected)

The first version combined the two marginals by averaging:

```
injection_probability = (harness_rate + model_rate) / 2
```

This was disclosed at the time as "a defensible approximation of a missing
joint statistic." Testing on `shard_0000_calls.parquet` (100% `claude_code`,
harness rate 0.2928) showed the achieved injection rate landing around 0.18,
not 0.29 — and per-model actual rates for both `DeepSeek-V3.2` and
`Kimi-K2.5` also missed their marginal targets by 12+ percentage points.

Claude Code's analysis confirmed this was **not** small-sample noise: harness
rates and model rates sit on very different scales (claude_code is 0.2928;
typical model rates are 0.02-0.07). Averaging two numbers on different scales
always regresses the result toward their midpoint, for any sample size —
more data would not have closed the gap. The averaging approach was a
structural bug, not a labeling problem.

## Decision (revised)

Replaced averaging with a harness-rate-anchored, model-relative-risk
multiplier:

```
mean_model_rate = mean(model_failure_rates.values())
model_relative_risk = model_failure_rates[row.model] / mean_model_rate
injection_probability = harness_failure_rates[row.harness] * model_relative_risk
injection_probability = min(injection_probability, 0.95)  # sanity cap
```

`harness_rate` is the anchor — the primary, operationally meaningful signal
(how often does this harness fail, in absolute terms). `model_relative_risk`
is a multiplier, not a second absolute rate: a model exactly at the average
model failure rate leaves the harness rate unchanged (multiplier = 1); a
worse-than-average model pushes it up; a better-than-average model pulls it
down. The result stays anchored to the harness's real scale instead of being
pulled toward the midpoint of two incompatible scales. The 0.95 cap guards
against a high-harness-rate x high-relative-risk combination pushing the
probability unrealistically close to certainty.

Re-tested on `shard_0000_calls.parquet`: actual injection rates now land
within ~1 percentage point of the corrected expected value (harness_rate ×
that group's actual mix of model_relative_risk) for both the harness as a
whole and each model individually — well inside the 5pp small-sample-variance
threshold, versus 11-12pp drift under the averaging approach.

## Decision (second revision): bound the multiplier, not just the result

The anchor+multiplier fix above still had a real gap. `claude-opus-4-5`'s
rate (0.2294) is ~3.0x `mean_model_rate` (0.0762). Paired with `claude_code`
(0.2928), the unbounded multiplier gives `0.2928 * 3.0105 ≈ 0.88` — meaning
~88% of that model's calls under that harness would get a synthetic failure
injected, far beyond anything in the documented statistics, and close enough
to the 0.95 cap that it wasn't a safe margin. Capping only the final
probability doesn't fix this: it clips the symptom (the rare cases that
actually hit 0.95) while leaving every below-cap case for that model, like
this one, still inflated by the raw ~3.0x multiplier.

Fix: clip `model_relative_risk` itself to `[0.5, 2.0]` *before* using it as
a multiplier, in addition to keeping the final-probability cap at 0.95 as a
backstop:

```
mean_model_rate = mean(model_failure_rates.values())
model_relative_risk = clip(model_failure_rates[row.model] / mean_model_rate, 0.5, 2.0)
injection_probability = harness_failure_rates[row.harness] * model_relative_risk
injection_probability = min(injection_probability, 0.95)  # backstop, not primary fix
```

Re-tested on `shard_0004_calls.parquet` (100% `claude_code`, 2935
`claude-opus-4-5` rows — the exact high-harness-rate x outlier-model
combination this targets):

- `claude-opus-4-5`: raw relative risk 3.0105 -> clipped to 2.0000.
  Achieved injection rate **59.69%**, matching the corrected expected value
  of 58.56% (`0.2928 * 2.0`), versus the ~88% the unclipped multiplier would
  have produced. Not close to the 0.95 cap.
- The lower bound also engaged for low-risk models in the same shard:
  `gemini-3-pro-preview` (raw relative risk 0.3084) and `gpt-5.2-2025-12-11`
  (raw relative risk 0.0, since its documented rate is 0.0) both clipped up
  to 0.5, so neither is treated as functionally failure-proof under a
  high-failure harness.

## Consequences

- Injection probability is still an approximation (the true joint rate isn't
  published), but it no longer has a built-in bias that persists regardless
  of sample size, and no single model's marginal rate — however far from
  `mean_model_rate` — can swing the combined probability more than 2x up or
  0.5x down from the harness's own rate.
- The reporting in `inject_synthetic_retries` compares actual achieved rates
  against the *expected value implied by the row-level formula* (mean of
  each row's own `harness_rate * clipped model_relative_risk` within a
  group), not against the raw marginal alone — the raw marginal and the
  unclipped relative risk are both printed for reference, since a harness's
  true expected rate depends on the mix of models it pairs with in the data.
- The `[0.5, 2.0]` bound is a chosen sanity range, not a documented dataset
  statistic — same disclosure principle as the anchor+multiplier design
  itself: an explicit, defensible approximation, not a fabricated precision.

## Known limitation: validation scope

The actual-vs-expected drift checks performed throughout this build (per-shard,
then full-dataset) confirm that the RNG sampling correctly matches the computed
`injection_probability` values — i.e. that the code samples Bernoulli draws
correctly at scale. That is an engineering-correctness claim, not a model
one. It does **not** validate that the harness-anchored, clipped-relative-risk
formula itself is a realistic model of how real-world LLM API failures
actually behave — that is a business/statistical judgment no amount of
drift-checking against the formula's own output can establish, since the
"expected" value being checked against is derived from the same formula being
tested. These are two different claims, and only the first has been tested.

## Timestamp parsing bug

The full 9-shard `inject_all_shards()` run initially crashed inside
`pd.to_datetime()` on `start_time`/`end_time`. A handful of real timestamps
in `shard_0003` and `shard_0005` land on exactly zero microseconds and are
recorded without a fractional-seconds suffix (e.g. `...T16:09:33+00:00`
instead of `...T16:09:33.000000+00:00`), which pandas' format-inference
can't parse in the same pass as the microsecond-bearing majority. Fixed with
explicit `format="ISO8601"`, which handles both shapes. This was only caught
by running the full pipeline end-to-end across all 9 shards — the earlier
single-shard testing (`shard_0000`, `shard_0001`, `shard_0004`) happened not
to contain that edge case, so it passed cleanly despite the latent bug.

## Known limitation: pricing accuracy is a separate gap from injection-formula validity

`config/pricing.yaml` (`src/pricing.py`) has its own, distinct limitation from
the "Known limitation: validation scope" section above — worth stating
explicitly so the two don't get conflated as "the same disclosed
approximation" when they're not:

- **Injection-formula validity** (this ADR, above) is about whether the
  harness-anchored, clipped-relative-risk *formula* realistically models how
  real-world LLM failures behave. No amount of testing inside this codebase
  can resolve that — it's a modeling judgment call, checked against the
  formula's own output, not against independent ground truth.
- **Pricing accuracy** is a *sourcing* problem, not a modeling one: it's
  about whether the dollar figures in `pricing.yaml` are the right number for
  a well-defined question ("what does this exact model, billed through this
  exact vendor, actually cost per token"). Unlike the injection formula, this
  one *is* independently checkable against the real world, and one instance
  of it being wrong was caught and fixed directly: `pricing.yaml` originally
  priced `DeepSeek-V3.2`, `Kimi-K2.5`, and `gpt-5.2-2025-12-11` using each
  model's own direct vendor API pricing, but the dataset's own `provider`
  column (`attributes.gen_ai.provider.name`) shows all three are actually
  billed through `azure.ai.openai` — a different vendor with its own rate
  card. Corrected by researching Azure AI Foundry/Azure OpenAI Service
  pricing specifically; total cost changed by +6.29% ($31,670.37 ->
  $33,663.00) as a result. Even after that correction, `pricing.yaml`'s own
  `pricing_source_note` per model still discloses real remaining gaps
  (third-party trackers instead of vendor pages, a retired model's
  last-known price, Azure's rate card lagging a recent OpenAI price cut) —
  those are sourcing-confidence caveats, not injection-formula-style
  business judgment calls.
