# Ledgr

Ledgr turns raw AI agent execution traces into cost-per-successful-outcome
data: for a given harness (agent framework) and model, how much does it
actually cost — in dollars, including the price of failed attempts and
retries — to get one successful task done? Most agent benchmarking looks at
accuracy or latency in isolation; Ledgr's core metric is spend, attributed
down to the individual call, so failure and retry cost stop being invisible
line items and become a number you can compare across harnesses and models.

## Data source

The base data is [Exgentic/agent-llm-traces-v2](https://huggingface.co/datasets/Exgentic/agent-llm-traces-v2)
(`data/raw/*.parquet`, 9 shards) — real agent session traces with nested
OpenTelemetry spans per LLM call.

The source dataset's own publishing pipeline filters out failed
retries before publication, so the raw traces only show the calls that
ultimately succeeded — retry overhead, the thing Ledgr is built to measure,
isn't present in the data as published. To make cost-of-failure visible
at all, this pipeline injects *synthetic* failed-retry spans, calibrated to
the dataset card's own documented per-harness and per-model failure rates
(`config/failure_rates.yaml`), rather than leaving retry cost as an unmeasured
gap. Every synthetic row is flagged (`is_synthetic_retry=True`) and clearly
distinguishable from a real call — nothing is presented as real that isn't.
The full design history of that calibration (including two rounds of
correcting a real bias in the formula) is in
[`docs/adr/0001-synthetic-retry-injection-calibration.md`](docs/adr/0001-synthetic-retry-injection-calibration.md).

## Pipeline overview

| Stage | Script | Output |
|---|---|---|
| 1. Raw shards | `data/raw/*.parquet` | 9 shards of nested session traces (input, not generated) |
| 2. Exploded calls | `src/explode_spans.py` | `data/processed/shard_*_calls.parquet` — one row per real LLM call, flattened out of the nested `spans` arrays |
| 3. Synthetic retry injection | `src/inject_retries.py` | `data/processed/shard_*_augmented.parquet` — real calls + calibrated synthetic failed-retry rows |
| 4. Cost calculation | `src/pricing.py` | `data/processed/shard_*_final.parquet` — adds `execution_cost_usd` per row; **this is the final output, ready for upload** |

## How to run it

```bash
# from the repo root
source venv/bin/activate
pip install -r requirements.txt

# stage 2: explode raw shards into flat call-level tables
python3 src/explode_spans.py

# stage 3: inject calibrated synthetic retries
python3 src/inject_retries.py

# stage 4: calculate cost per call, write the final output
python3 src/pricing.py

# run the test suite
pytest
```

Each stage reads the previous stage's output from `data/processed/` and is
safe to re-run independently once its input exists — `data/raw/` and
`data/processed/` are both gitignored (large generated/downloaded files, not
source), so a fresh checkout needs `data/raw/*.parquet` populated from the
Exgentic dataset before stage 2 will find anything to process.

Each stage processes shards one at a time rather than loading all 9 into
memory at once — the raw `spans` column alone is large enough to OOM-kill a
16GB machine if concatenated across all shards before processing (see the
ADR for the numbers).

## Known limitations

Pulled directly from what's already disclosed in
[`docs/adr/0001-synthetic-retry-injection-calibration.md`](docs/adr/0001-synthetic-retry-injection-calibration.md)
and `config/pricing.yaml` — stated here plainly rather than left buried in
those files:

- **The injection formula is unvalidated as a model of reality, and can't be
  validated from inside this codebase.** The actual-vs-expected drift checks
  throughout this build confirm the code correctly samples from the
  probabilities it computes (engineering correctness) — they do **not**
  confirm the harness-anchored, clipped-relative-risk formula itself
  realistically models how real-world LLM API failures behave (a
  business/statistical judgment call). These are different claims; only the
  first has been tested.
- **Small-sample drift is expected and disclosed, not a bug.** Per-shard
  actual-vs-expected injection rates can drift up to several percentage
  points from what the row-level formula predicts, purely from sample-size
  variance — the full-dataset aggregate consistently lands under 0.3pp
  drift, confirming it. Any per-shard report that flags drift >5pp is
  telling you the sample is small, not that something is broken.
- **Pricing confidence varies per model, and 3 of 5 models needed a
  billing-vendor correction.** `config/pricing.yaml` prices each model at the
  rate of whoever actually bills for it (per the data's own `provider`
  field), not whoever trained it — `DeepSeek-V3.2`, `Kimi-K2.5`, and
  `gpt-5.2-2025-12-11` are billed via Azure, not each model-maker's own
  direct API, and were re-priced accordingly. Per-model confidence:
  - **High**: `claude-opus-4-5` (Anthropic's own announcement),
    `gemini-3-pro-preview` (exact name match across trackers, though the
    >200K-context tier isn't modeled — every row uses the standard rate).
  - **Moderate-high**: `Kimi-K2.5` (Azure rate independently confirmed to
    match Moonshot's own list price).
  - **Moderate**: `gpt-5.2-2025-12-11` (Azure's rate card appears to lag a
    recent OpenAI direct-API price cut), `DeepSeek-V3.2` (Azure AI Foundry
    third-party tracker, not fetched from Microsoft's own page directly; the
    underlying model was also retired mid-project).
  - This pricing gap is a distinct issue from the injection-formula gap
    above — one is a modeling judgment call, the other was a concrete
    sourcing error that got checked against the data and fixed. See the
    ADR's "pricing accuracy is a separate gap" section for why they
    shouldn't be conflated.
- **`agent_cost` (source dataset) and `execution_cost_usd` (this pipeline)
  don't reconcile, and aren't meant to.** They're independently computed,
  at different granularities (session-level vs. per-call), with different
  pricing (the source dataset appears to use direct-vendor rates even for
  Azure-billed models; this pipeline corrects for that). See
  [`docs/data_dictionary.md`](docs/data_dictionary.md) for the verified
  numbers behind this.
