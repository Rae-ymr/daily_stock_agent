"""
FastAPI wrapper around the agent pipeline.

Run standalone:
    uvicorn app.main:app --reload
Then visit http://127.0.0.1:8000/docs for the interactive API explorer.
"""

import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agent.graph import run_pipeline
from app.heatmap import build_heatmap_html
from data.ingest_prices import fetch_price_history

app = FastAPI(title="Daily Stock Analysis Agent")


class ChatRequest(BaseModel):
    ticker: str


@app.post("/chat")
def chat(req: ChatRequest):
    """Runs the full pipeline for a given ticker and returns the result."""
    # NOTE: run_pipeline currently has a blocking input() call in
    # human_checkpoint_node — you'll want to replace that with a
    # proper two-step API flow (submit draft -> separate approve
    # endpoint) before this works well over HTTP.
    result = run_pipeline(req.ticker)
    return result


@app.get("/heatmap", response_class=HTMLResponse)
def heatmap():
    """Renders a diverging-color heatmap of % change for each tracked ticker."""
    tickers = [t.strip() for t in os.getenv("STOCK_LIST", "AAPL,MSFT").split(",") if t.strip()]
    cells = []
    for ticker in tickers:
        rows = fetch_price_history(ticker, period="5d")["rows"]
        if not rows:
            continue
        first, last = rows[0], rows[-1]
        pct_change = (last["close"] - first["close"]) / first["close"] * 100
        cells.append({"ticker": ticker, "pct_change": pct_change, "close": last["close"]})
    return HTMLResponse(build_heatmap_html(cells))


@app.get("/eval")
def eval_endpoint():
    """TODO: expose eval/run_eval.py results here, or trigger a fresh run."""
    return {"status": "not implemented yet"}


@app.get("/health")
def health():
    return {"status": "ok"}
