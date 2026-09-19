# Daily Stock Analysis Agent

A corrective-RAG agent that ingests price + news data for a ticker, retrieves and
self-corrects on relevant context, drafts a summary/recommendation, and stops for
human approval before anything is logged or sent out.

This repo is a **scaffold** — every module has real function signatures and
docstrings, but the actual logic (marked `TODO` / `raise NotImplementedError`)
is left for you to fill in as you learn each piece. Build and test each file
on its own before wiring it into the full pipeline.

## Project structure

```
daily-stock-agent/
├── data/
│   ├── ingest_prices.py      # Pulls price history via yfinance
│   └── ingest_news.py        # Pulls news headlines (plug in your API)
├── agent/
│   ├── retrieval.py           # Corrective RAG: retrieve, grade, retry — BUILD THIS FIRST
│   ├── tools.py                 # Agent tools (price summary, news lookup)
│   ├── memory.py               # Conversation history
│   └── graph.py                 # Wires all stages into the full pipeline
├── eval/
│   ├── test_set.json          # Your labeled question/answer examples
│   └── run_eval.py             # Scores the agent against test_set.json
├── app/
│   └── main.py                  # FastAPI wrapper (/chat, /eval, /health)
├── requirements.txt
├── .env.example                # Copy to .env and fill in your keys
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Setup

```bash
cd daily-stock-agent
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
# now edit .env with your real API keys
```

## How to run each file, in the order you'll build them

### 1. `data/ingest_prices.py` — no setup needed, works out of the box
```bash
python -m data.ingest_prices AAPL
```
Should print the last row of price data for AAPL. This one already works;
it's your baseline for "is my environment set up correctly."

### 2. `data/ingest_news.py` — needs `NEWS_API_KEY` in `.env`
```bash
python -m data.ingest_news AAPL
```
Without a key it prints placeholder data so you can still test downstream
code. Fill in the real API call (NewsAPI, Tavily, GNews — your choice) when
you're ready.

### 3. `agent/retrieval.py` — build this first, it's the core RAG logic
```bash
python -m agent.retrieval "What happened to AAPL this week?"
```
Currently just prints a reminder — you need to fill in `build_vector_store`,
`retrieve`, `grade_relevance`, and `rewrite_query` before this does anything.
Test each of those four functions individually (e.g. in a Python shell)
before trying to run the whole retry loop.

### 4. `agent/tools.py` — depends on files 1 and 2
```bash
python -m agent.tools AAPL
```
Prints a price summary and recent news for AAPL. This confirms your data
modules are wired correctly before the agent tries to use them.

### 5. `agent/memory.py` — standalone, no dependencies
```bash
python -m agent.memory
```
Just a sanity check that the memory object stores and returns messages
correctly.

### 6. `agent/graph.py` — the full pipeline, needs 1–5 filled in
```bash
python -m agent.graph AAPL
```
Runs all six pipeline stages in sequence. Will raise `NotImplementedError`
at `agent_reasoning_node` and `draft_summary_node` until you fill those in
with a real LLM call. The `human_checkpoint_node` will pause for a y/n input
in your terminal — that's intentional, it's your approval gate.

### 7. `eval/run_eval.py` — run once `agent/graph.py` produces real answers
```bash
python -m eval.run_eval
```
Loads `eval/test_set.json`, runs each case's `ticker` through the pipeline
up to `draft_summary_node` (skipping the human-approval step, which isn't
meaningful in an automated eval), and scores the draft against
`expected_answer` via embedding cosine similarity. Cases still holding the
placeholder `expected_answer` text are skipped, not scored as 0 — fill in
real expected answers in `test_set.json` once you've seen real drafts to
know what a good answer looks like.

### 8. `app/main.py` — wraps everything behind an API
```bash
uvicorn app.main:app --reload
```
Then open `http://127.0.0.1:8000/docs` to see and test the `/chat`, `/eval`,
and `/health` endpoints interactively. Note: the human-checkpoint step
currently blocks on terminal input, which won't work well over HTTP — you'll
want to split it into a "submit draft" + "approve draft" endpoint pair once
you get here (that's a good Week 4 productionizing task).

### 9. Docker — run everything in one command, once the app works locally
```bash
docker compose up --build
```
Same endpoints as above, now containerized.

## Suggested build order

Match this to your study plan: `ingest_prices.py` → `ingest_news.py` →
`retrieval.py` (corrective RAG) → `tools.py` → `graph.py`'s reasoning/draft
nodes (fine-tuned model or plain LLM call) → `run_eval.py` → `main.py` →
Docker. Each arrow is roughly "this week's new concept, applied to this
file."
