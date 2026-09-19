"""
FastAPI wrapper around the agent pipeline.

Run standalone:
    uvicorn app.main:app --reload
Then visit http://127.0.0.1:8000/docs for the interactive API explorer.
"""

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agent.graph import (
    decision_node,
    draft_summary_node,
    ingest_node,
    intel_node,
    log_and_notify_node,
    retrieve_and_grade_node,
    risk_node,
    technical_node,
)
from agent.retrieval import build_vector_store
from app.heatmap import build_heatmap_html
from app.session_store import create_session, delete_session, load_session, update_session
from data.ingest_prices import fetch_price_history

# Mirrors the revision cap in agent/graph.py's route_after_checkpoint —
# reject twice and the pipeline finalizes as rejected instead of
# drafting a third time.
MAX_REVISIONS = 2

app = FastAPI(title="Daily Stock Analysis Agent")


class AnalyzeRequest(BaseModel):
    ticker: str


class ApproveRequest(BaseModel):
    approved: bool


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """
    Runs the pipeline through draft_summary_node and stops there — the API
    equivalent of the first half of human_checkpoint_node ("show the
    draft"). The draft is stashed in Redis under a session_id; call
    POST /approve/{session_id} with the human's decision to continue.
    """
    state = {"ticker": req.ticker}
    state.update(ingest_node(state))
    state["store"] = build_vector_store(state["news"])
    state.update(retrieve_and_grade_node(state))
    state.update(technical_node(state))
    state.update(intel_node(state))
    state.update(risk_node(state))
    state.update(decision_node(state))
    state.update(draft_summary_node(state))

    session_id = create_session(
        {
            "ticker": state["ticker"],
            "price_summary": state.get("price_summary", ""),
            "analysis": state.get("analysis", ""),
            "draft": state["draft"],
            "revision_count": 0,
        }
    )
    return {
        "session_id": session_id,
        "ticker": req.ticker,
        "decision": state.get("decision"),
        "draft": state["draft"],
    }


@app.post("/approve/{session_id}")
def approve(session_id: str, req: ApproveRequest):
    """
    The API equivalent of the second half of human_checkpoint_node
    ("approve or reject"). Approval (or hitting MAX_REVISIONS) finalizes
    via log_and_notify_node; rejection under the cap redrafts and waits
    for another call to this same endpoint.
    """
    state = load_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    if req.approved:
        log_and_notify_node({"ticker": state["ticker"], "approved": True, "draft": state["draft"]})
        delete_session(session_id)
        return {"session_id": session_id, "status": "final", "approved": True, "draft": state["draft"]}

    revision_count = state["revision_count"] + 1
    if revision_count >= MAX_REVISIONS:
        log_and_notify_node({"ticker": state["ticker"], "approved": False, "draft": state["draft"]})
        delete_session(session_id)
        return {"session_id": session_id, "status": "final", "approved": False, "draft": state["draft"]}

    redraft = draft_summary_node(
        {
            "ticker": state["ticker"],
            "price_summary": state["price_summary"],
            "analysis": state["analysis"],
        }
    )
    state["draft"] = redraft["draft"]
    state["revision_count"] = revision_count
    update_session(session_id, state)
    return {
        "session_id": session_id,
        "status": "pending",
        "revision_count": revision_count,
        "draft": state["draft"],
    }


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
