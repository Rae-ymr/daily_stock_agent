"""
Wires the pipeline stages together as a LangGraph StateGraph:
  ingest -> retrieve_and_grade -> agent_reasoning -> draft_summary
  -> human_checkpoint -> approved: log_and_notify
                          rejected: back to draft_summary (up to 2 revisions)

Fill in:
  - agent_reasoning_node (needs an LLM client)
  - draft_summary_node (needs an LLM client)

Run standalone (once agent_reasoning_node/draft_summary_node are filled in):
    python -m agent.graph AAPL
"""

import os
import smtplib
import sys
from email.mime.text import MIMEText
from typing import Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from agent.retrieval import retrieve_with_retry
from agent.tools import get_price_summary, get_recent_news

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
_tools = [get_price_summary, get_recent_news]
_tools_by_name = {t.name: t for t in _tools}
_llm_with_tools = _llm.bind_tools(_tools)
MAX_TOOL_ROUNDS = 3


class AgentState(TypedDict):
    ticker: str
    store: Optional[object]
    price_summary: str
    news: list[dict]
    retrieved_context: list[dict]
    analysis: str
    draft: str
    approved: bool
    revision_count: int


def ingest_node(state: AgentState) -> dict:
    """Stage 1: pull price + news data for the ticker."""
    ticker = state["ticker"]
    return {
        "price_summary": get_price_summary.invoke({"ticker": ticker}),
        "news": get_recent_news.invoke({"ticker": ticker}),
    }


def retrieve_and_grade_node(state: AgentState) -> dict:
    """Stage 2: corrective RAG over the news/documents."""
    store = state.get("store")
    if store is None:
        return {"retrieved_context": []}
    query = f"What's driving {state['ticker']} recently?"
    return {"retrieved_context": retrieve_with_retry(query, store)}


def agent_reasoning_node(state: AgentState) -> dict:
    """
    Stage 3: lets the LLM reason over price_summary + retrieved_context.
    It can call get_price_summary/get_recent_news again (e.g. for a
    different ticker mentioned in the news) if it decides the context
    it already has isn't enough, up to MAX_TOOL_ROUNDS.
    """
    context = "\n".join(
        f"- {c['content']}" for c in state.get("retrieved_context", [])
    )
    messages = [
        SystemMessage(
            "You are a financial analysis assistant preparing notes for a "
            "daily stock summary. Use the price and news context given. "
            "Call a tool only if that context is missing something you need."
        ),
        HumanMessage(
            f"Ticker: {state['ticker']}\n"
            f"Price summary: {state.get('price_summary', 'unavailable')}\n"
            f"Recent news context:\n{context or 'none retrieved'}\n\n"
            "Analyze what's driving this stock and note anything uncertain."
        ),
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        response = _llm_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return {"analysis": response.content}
        for call in response.tool_calls:
            result = _tools_by_name[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return {"analysis": messages[-1].content}


def draft_summary_node(state: AgentState) -> dict:
    """Stage 4: produces the actual draft text, using the analysis notes."""
    prompt = (
        f"Ticker: {state['ticker']}\n"
        f"Price summary: {state.get('price_summary', 'unavailable')}\n"
        f"Analysis notes: {state.get('analysis', 'none')}\n\n"
        "Write a short (3-5 sentence) daily summary: what happened and "
        "what to watch next. This is informational only, not a buy/sell "
        "recommendation. End with exactly: 'Status: draft — pending human review.'"
    )
    response = _llm.invoke(prompt)
    return {"draft": response.content}


def human_checkpoint_node(state: AgentState) -> dict:
    """
    Stage 5: for now, a CLI prompt is fine. Show the draft, ask for
    approve/edit/reject before it's considered final.
    """
    print("\n--- DRAFT ---")
    print(state.get("draft", "<no draft produced>"))
    decision = input("Approve? [y/n]: ")
    approved = decision.lower().startswith("y")
    return {
        "approved": approved,
        "revision_count": state.get("revision_count", 0) + (0 if approved else 1),
    }


def route_after_checkpoint(state: AgentState) -> str:
    """Conditional edge: redraft on rejection (up to 2 revisions), else log."""
    if state.get("approved"):
        return "log_and_notify"
    if state.get("revision_count", 0) >= 2:
        print("Max revisions reached — logging as rejected.")
        return "log_and_notify"
    return "draft_summary"


def send_notification_email(ticker: str, draft: str) -> None:
    """
    Emails the approved draft via Gmail SMTP. Requires SMTP_USER (a Gmail
    address) and SMTP_APP_PASSWORD (a Gmail app password, not your regular
    password — generate one at myaccount.google.com/apppasswords) in .env.
    Recipient defaults to NOTIFY_EMAIL_TO if set, else the hardcoded address.
    """
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_APP_PASSWORD")
    if not smtp_user or not smtp_password:
        print("SMTP_USER/SMTP_APP_PASSWORD not set — skipping email, printing instead:")
        print(draft)
        return

    to_addr = os.getenv("NOTIFY_EMAIL_TO", "yumingrui.silverdew@gmail.com")
    msg = MIMEText(draft)
    msg["Subject"] = f"Daily stock summary: {ticker}"
    msg["From"] = smtp_user
    msg["To"] = to_addr

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, [to_addr], msg.as_string())
    print(f"Notification emailed to {to_addr}")


def log_and_notify_node(state: AgentState) -> dict:
    """Stage 6: logs the outcome and emails the draft if it was approved."""
    print(f"Logged. Approved: {state.get('approved')}")
    if state.get("approved"):
        send_notification_email(state["ticker"], state.get("draft", ""))
    return {}


def build_graph():
    """Assembles and compiles the StateGraph. Call once, reuse the result."""
    graph = StateGraph(AgentState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("retrieve_and_grade", retrieve_and_grade_node)
    graph.add_node("agent_reasoning", agent_reasoning_node)
    graph.add_node("draft_summary", draft_summary_node)
    graph.add_node("human_checkpoint", human_checkpoint_node)
    graph.add_node("log_and_notify", log_and_notify_node)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "retrieve_and_grade")
    graph.add_edge("retrieve_and_grade", "agent_reasoning")
    graph.add_edge("agent_reasoning", "draft_summary")
    graph.add_edge("draft_summary", "human_checkpoint")
    graph.add_conditional_edges(
        "human_checkpoint",
        route_after_checkpoint,
        {"draft_summary": "draft_summary", "log_and_notify": "log_and_notify"},
    )
    graph.add_edge("log_and_notify", END)

    return graph.compile()


_compiled_graph = None


def run_pipeline(ticker: str, store=None) -> dict:
    """Public entry point — used by app/main.py and eval/run_eval.py."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph.invoke({"ticker": ticker, "store": store, "revision_count": 0})


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    run_pipeline(ticker)
