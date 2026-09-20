# Daily Stock Analysis Agent

A multi-agent pipeline that ingests price + news data for a ticker, runs a
corrective-RAG retrieval step, then fans out to four independent parallel
agents — technical (price action), intel (news), quant (a trained LightGBM
classifier with an AutoARIMA forecast as one of its features), and risk
(insider activity, earnings/regulatory risk, valuation anomalies) — before
combining everything into a structured buy/hold/sell decision, and stopping
for human approval before anything is logged or sent out.

Every module below is implemented, not a stub — the one exception is
`agent/tools.py`'s `check_forecast_anomaly`, an optional forecasting/anomaly
hook left for later. See `docs/multi_agent_architecture.md` for how the
technical/intel/quant/risk/decision agents fit together, and
`docs/human_checkpoint_flow.md` for the design of the human-approval step
(CLI `input()` vs. the API's `/analyze` + `/approve` split).

## Project structure

```
daily-stock-agent/
├── data/
│   ├── ingest_prices.py      # Price history + fundamentals (P/E, P/B) + technical
│   │                             indicators (MA crossover, RSI, MACD, Bollinger, volume) via yfinance
│   └── ingest_news.py        # Pulls news headlines via Tavily
├── agent/
│   ├── llm.py                    # Picks the chat LLM (Groq default, OpenAI optional)
│   ├── retrieval.py           # Corrective RAG: retrieve, grade, retry
│   ├── tools.py                 # Price summary + news lookup tools, used by ingest_node
│   ├── memory.py               # Conversation history
│   └── graph.py                 # LangGraph pipeline: ingest → retrieve/grade →
│                                   [technical, intel, quant, risk] (parallel)
│                                   → decision → draft → human checkpoint → notify
├── ml/
│   ├── autoarima_forecast.py   # 5-day return forecast via Nixtla's AutoARIMA
│   ├── features.py               # Shared feature-building — same function trains and serves
│   ├── train.py                   # Builds a pooled panel across STOCK_LIST, trains LightGBM
│   └── predict.py                 # Loads the trained model, scores one ticker (used by quant_node)
├── eval/
│   ├── test_set.json          # Labeled ticker/question/expected_answer cases
│   └── run_eval.py             # Runs the pipeline per case, scores by embedding similarity
├── app/
│   ├── main.py                  # FastAPI wrapper (/analyze, /approve, /heatmap, /eval, /health)
│   ├── session_store.py        # Redis-backed session state between /analyze and /approve
│   └── heatmap.py               # Renders the /heatmap HTML
├── docs/
│   ├── multi_agent_architecture.md # technical/intel/risk/decision agent design
│   └── human_checkpoint_flow.md   # CLI vs. API flow for the human-approval step
├── requirements.txt
├── .env.example                # Copy to .env and fill in your keys
├── Dockerfile
├── docker-compose.yml          # agent + redis services
└── README.md
```

## Setup

```bash
cd daily-stock-agent
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
# now edit .env with your real API keys (GROQ_API_KEY or OPENAI_API_KEY, TAVILY_API_KEY)
# RISK_OVERRIDE_ENABLED defaults to true — leave it unless you're debugging risk_node
```

## How to run each piece

### 1. `data/ingest_prices.py` — no setup needed
```bash
python -m data.ingest_prices AAPL
```
Prints the last row of price data for AAPL via yfinance. Your baseline for
"is my environment set up correctly."

### 2. `data/ingest_news.py` — needs `TAVILY_API_KEY` in `.env`
```bash
python -m data.ingest_news AAPL
```
Without a key it prints placeholder data so downstream code still runs.
Get a free key at tavily.com.

### 3. `agent/retrieval.py` — the corrective-RAG core
```bash
python -m agent.retrieval AAPL
```
Fetches news for the ticker, embeds it locally (sentence-transformers) into
an in-memory Chroma store, then runs `retrieve_with_retry`: retrieve → grade
relevance with an LLM → if weak, rewrite the query and retry (up to
`MAX_RETRIES`) → return the best result found.

### 4. `agent/tools.py` — depends on files 1 and 2
```bash
python -m agent.tools AAPL
```
Prints a price summary and recent news for AAPL, wrapped as LangChain
`@tool`s. `ingest_node` in `agent/graph.py` invokes them directly (not via
an LLM tool-calling loop — there isn't one in this pipeline; the `@tool`
wrapping is kept for interface consistency and in case a future node
needs the LLM to call them dynamically).

### 5. `agent/memory.py` — standalone, no dependencies
```bash
python -m agent.memory
```
Sanity check that the memory object stores and returns messages correctly.

### 6. `ml/train.py` — trains the quant signal (optional but recommended)
```bash
python -m ml.train
```
Pools price history across `STOCK_LIST` (5 years, sampled every 5th day),
builds a feature row per sample (technical indicators + an AutoARIMA
forecast — see `ml/features.py`), and trains a LightGBM classifier to
predict "will the 5-day forward return be positive?". Splits train/test
**by date**, not randomly (a random split would leak future rows into
training — see `docs/multi_agent_architecture.md`). Takes a few minutes
— AutoARIMA fits once per sampled row (an order search, not a lookup).
Saves to `ml/model.pkl` (gitignored; not committed). If you skip this
step, `agent/graph.py`'s `quant_node` still works — it just returns
`quant_signal=None`, which downstream nodes treat as "no opinion."

### 7. `agent/graph.py` — the full pipeline
```bash
python -m agent.graph AAPL
```
Runs the full pipeline: ingest → retrieve & grade → `technical` + `intel`
+ `quant` + `risk` (all four run in parallel — technical reasons over
price data only, intel over retrieved news only, quant runs the trained
LightGBM model from step 6, risk runs its own risk-focused search +
fundamentals check independently of the other three) → `decision`
(combines all four into a structured buy/hold/sell + price target via
`DecisionOutput`, with `risk`'s hard/soft flags enforced by
`apply_risk_override()` — see `docs/multi_agent_architecture.md`) → draft
summary → human checkpoint → log & notify. `human_checkpoint_node` pauses
for a y/n input in your terminal — that's the approval gate. Reject twice
and it finalizes as rejected instead of drafting a third time (see
`route_after_checkpoint`).

### 8. `eval/run_eval.py`
```bash
python -m eval.run_eval
```
Loads `eval/test_set.json`, runs each case's `ticker` through the pipeline
up to `draft_summary_node` (skipping human approval — not meaningful in an
automated eval), and scores the draft against `expected_answer` via
embedding cosine similarity. Cases still holding placeholder text are
skipped, not scored as 0.

### 9. `app/main.py` — wraps the pipeline behind an API
Needs Redis reachable (holds draft state between `/analyze` and `/approve`
— see `docs/human_checkpoint_flow.md`). Run one locally first:
```bash
redis-server &
uvicorn app.main:app --reload
```
Then open `http://127.0.0.1:8000/docs` to test the endpoints interactively:
- `POST /analyze {ticker}` — runs the pipeline through `draft_summary_node`,
  returns `{session_id, decision, draft}` (`decision` is the structured
  `DecisionOutput` JSON — buy/hold/sell, price target, rationale).
- `POST /approve/{session_id} {approved}` — approve to finalize, or reject
  to get a redraft (same revision cap as the CLI flow); call it again with
  the same `session_id` until it's approved or the cap is hit.
- `GET /heatmap` — diverging-color % change heatmap for `STOCK_LIST` tickers.
- `GET /eval` — not implemented yet; currently returns a placeholder status.
- `GET /health` — liveness check.

### 10. Docker
```bash
docker compose up --build
```
Runs `agent` and `redis` together; same endpoints as above.
