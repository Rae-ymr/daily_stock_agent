# Multi-agent architecture

How `agent/graph.py` turns one ticker into a buy/hold/sell call. For the
human-approval step specifically (CLI vs. API), see
`docs/human_checkpoint_flow.md` — this doc covers everything upstream of
that.

## Flow

```mermaid
flowchart TD
    A[ingest] --> B[retrieve_and_grade]
    B --> T["technical\n(price data only)"]
    B --> I["intel\n(retrieved news only)"]
    B --> Q["quant\n(LightGBM + AutoARIMA forecast)"]
    T --> R["risk\n(sees technical + intel + quant)"]
    I --> R
    Q --> R
    R --> DEC["decision\n(structured buy/hold/sell)"]
    DEC --> D[draft_summary]
    D --> E[human_checkpoint]
    E --> F[log_and_notify]
```

`technical`, `intel`, and `quant` run in the same LangGraph superstep —
genuinely parallel, not just adjacent in the code — because none of the
three depends on either of the others' output. `risk` fans in from all
three (it needs their reads to know what it might be overriding) before
`decision` runs.

## What each node is responsible for

| Node | Reads | Does not read | Produces |
|---|---|---|---|
| `technical_node` | `price_summary`, `technical_indicators` (MA crossover, RSI, MACD, Bollinger width, volatility, volume change) | news, market data | `technical_analysis` (free text) |
| `intel_node` | `retrieved_context` (corrective-RAG news) | price data | `intel_analysis` (free text) |
| `quant_node` | trained LightGBM model + live feature row (see below) | news, LLM reasoning | `quant_signal` (`{probability_up, features}`, or `None` if untrained) |
| `risk_node` | `technical_analysis`, `intel_analysis`, `quant_signal`, a risk-focused RAG query, fundamentals (P/E, P/B) | — | `risk_assessment` (`RiskAssessment`: list of `RiskFlag` + summary) |
| `decision_node` | all of the above | — | `decision` (structured `DecisionOutput`) + `analysis` (text, for `draft_summary_node`) |

`technical_node` and `intel_node` are deliberately scoped to *not* see
each other's domain — each is told explicitly not to reference the
other's kind of signal. `quant_node` is a different *kind* of signal
entirely — not an LLM call at all, see below. `decision_node` is where
all reads get combined, and it's told to surface technical/intel
disagreement rather than silently picking a side.

## `quant_node` — a trained model, not an LLM call

Every other node in this graph is an LLM reasoning over data. `quant_node`
is different: it calls a **trained LightGBM classifier** (`ml/train.py`,
`ml/predict.py`) that predicts the probability of a positive 5-day
forward return, using:
- the same technical indicators `technical_node` sees (MA spread, RSI,
  MACD, Bollinger width, volatility, volume change), plus
- a **forecast from AutoARIMA** (Nixtla's `statsforecast`,
  `ml/autoarima_forecast.py`) as one more feature — a classical
  statistical model's forecasted 5-day return, fed into LightGBM
  alongside the hand-built indicators rather than used on its own.

This mixes two genuinely different ML approaches: LightGBM gives a
labeled, evaluable classification structure (a real train/test split,
accuracy/AUC you can report); AutoARIMA contributes whatever extra
signal an independent statistical forecast adds on top of the classical
indicators — cheaply, since it fits per-series in milliseconds to a few
seconds rather than requiring a pretrained model download.

**Train/serve skew, concretely addressed**: `ml/features.py`'s
`build_feature_row()` is the *only* place features are computed, called
identically by `ml/train.py` (building a historical panel, one row per
sampled past trading day) and `quant_node` (one row for "today"). Two
separate implementations — one for training, one for serving — is
exactly how train/serve skew creeps in; using the same function
guarantees it can't happen here.

**Leakage, concretely addressed**: `ml/train.py` splits train/test by
date (`TRAIN_TEST_SPLIT_DATE`), not a random shuffle. A random split
would let rows *after* the split date train a model evaluated on rows
*before* it — the model would effectively be evaluated on data it
implicitly saw the future of. A time-based split forces the test set to
be strictly later than everything the model trained on, matching how
the model is actually used (predicting forward, never backward).

`quant_node` returns `quant_signal=None` (not an error) if
`python -m ml.train` hasn't been run yet — every downstream node treats
a missing quant signal as "no opinion," not a failure.

## Why this is a pipeline, not a supervisor

`build_graph()` wires every edge with `graph.add_edge(...)` — the
sequence is fixed in code, not decided at runtime by an LLM. The only
conditional routing in the whole graph is `route_after_checkpoint`, and
that branches on a human's approve/reject decision plus a revision
counter, not on a model deciding which agent to call next.

A **supervisor** pattern would have a dedicated LLM node deciding, per
run, which of `technical`/`intel`/`risk` to invoke and in what order,
with workers handing control back to the supervisor after each step.
This project doesn't need that: every run needs the same fixed set of
signals (price action, news, risk) in the same order, so a static graph
is cheaper, easier to test, and easier to debug (a failure is always
"stage X broke," never "the supervisor routed to the wrong agent").
Supervisor-style dynamic routing would earn its complexity if different
requests legitimately needed different subsets of agents — this project
doesn't have that variation.

## Final output — `DecisionOutput`

```python
class DecisionOutput(BaseModel):
    decision: Literal["buy", "hold", "sell"]
    price_target: Optional[float]
    rationale: str
```

Produced via `with_structured_output`, not parsed out of free text — the
buy/hold/sell signal and price target are always machine-readable.
`state["decision"]` (this, as a dict) is what `POST /analyze` returns
alongside the draft.

## Risk override — two-level severity

`risk_node` can emit `RiskFlag`s with `severity` of:
- **soft** — downgrades a `buy` to `hold` and adds a visible warning to
  `rationale`.
- **hard** — vetoes a `buy` down to `hold` outright, *if*
  `RISK_OVERRIDE_ENABLED` (env var, default `true`).

This is enforced by `apply_risk_override()` **after** `decision_node`'s
LLM call, not inside the prompt — the override is guaranteed by code,
not something the model is merely asked to remember to apply. Non-buy
decisions (`hold`/`sell`) pass through unchanged; there's nothing to
downgrade or veto in those.

## Known simplifications (staged — not gaps to be surprised by)

- `technical_node` uses real indicators now (MA crossover, RSI, MACD,
  Bollinger Band width, 20-day volatility, volume change — computed in
  `data/ingest_prices.py`'s `fetch_technical_indicators`, not by the
  LLM). Support/resistance levels specifically are still not computed —
  only the crossover/RSI/MACD/volatility signals above.
- `intel_node` only searches single-ticker news; no SPY/QQQ market-wide
  context yet.
- `risk_node`'s insider-activity and lock-up-expiration checks ride on
  the same general news-embedding search as `intel_node` — if a risk
  never showed up in that search, it won't be flagged. A production
  version would want a dedicated feed (e.g. SEC Form 4 filings) instead
  of hoping the risk surfaces in a news search.
- `ml/train.py` pools only `STOCK_LIST` (2 tickers by default) over 5
  years, sampled every 5th day — a real quant setup would train on far
  more tickers and more history. `ml/autoarima_forecast.py` runs
  AutoARIMA with default settings (no explicit seasonal period, no
  exogenous regressors) — a reasonable default for daily stock closes,
  but a real deployment might tune the search space or add regressors
  (e.g. sector index level) if it improved held-out accuracy.
