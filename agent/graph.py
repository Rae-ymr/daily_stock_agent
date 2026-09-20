"""
Wires the pipeline stages together as a LangGraph StateGraph:
  ingest -> retrieve_and_grade -> [technical, intel, quant, risk] (parallel)
  -> decision -> draft_summary -> human_checkpoint
  -> approved: log_and_notify
     rejected: back to draft_summary (up to 2 revisions)

technical_node, intel_node, quant_node, and risk_node run in the same
superstep, fanning out from retrieve_and_grade and fanning back in at
decision_node — see build_graph().

Run standalone:
    python -m agent.graph AAPL
"""

import os
import smtplib
import sys
from email.mime.text import MIMEText
from typing import Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from agent.llm import get_chat_llm
from agent.retrieval import retrieve_with_retry
from agent.tools import get_price_summary, get_recent_news
from data.ingest_prices import fetch_fundamentals, fetch_technical_indicators
from ml.predict import predict_5d_up_probability

_llm = get_chat_llm()

# Hard risk flags veto a buy outright only when this is true — a kill
# switch in case the override logic itself misbehaves and needs to be
# disabled without a code change.
RISK_OVERRIDE_ENABLED = os.getenv("RISK_OVERRIDE_ENABLED", "true").lower() == "true"


class PriceRange(BaseModel):
    low: float
    high: float


# Ordered bearish -> bullish so apply_risk_override can shift a decision
# by index rather than special-casing every level.
DECISION_LEVELS = ["strong_sell", "sell", "hold", "buy", "strong_buy"]


class DecisionOutput(BaseModel):
    """Structured final call from decision_node — kept out of free text so
    the rating and price ranges are always machine-readable, not something
    scraped out of prose."""

    decision: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]
    buy_range: Optional[PriceRange] = Field(
        default=None, description="Price range worth considering as an entry, if the analysis supports one"
    )
    sell_range: Optional[PriceRange] = Field(
        default=None, description="Price range worth considering as an exit/take-profit, if the analysis supports one"
    )
    rationale: str


class RiskFlag(BaseModel):
    """A single risk finding from risk_node."""

    severity: Literal["soft", "hard"]
    category: Literal[
        "insider_activity",
        "earnings_warning",
        "regulatory_action",
        "valuation_anomaly",
        "lockup_expiration",
        "other",
    ]
    detail: str


class RiskAssessment(BaseModel):
    flags: list[RiskFlag] = Field(default_factory=list)
    summary: str


class AgentState(TypedDict):
    ticker: str
    store: Optional[object]
    price_summary: str
    technical_indicators: dict
    news: list[dict]
    retrieved_context: list[dict]
    technical_analysis: str
    intel_analysis: str
    quant_signal: Optional[dict]
    risk_assessment: Optional[dict]
    decision: Optional[dict]
    analysis: str
    draft: str
    approved: bool
    revision_count: int


def ingest_node(state: AgentState) -> dict:
    """Stage 1: pull price + technical indicators + news data for the ticker."""
    ticker = state["ticker"]
    return {
        "price_summary": get_price_summary.invoke({"ticker": ticker}),
        "technical_indicators": fetch_technical_indicators(ticker),
        "news": get_recent_news.invoke({"ticker": ticker}),
    }


def retrieve_and_grade_node(state: AgentState) -> dict:
    """Stage 2: corrective RAG over the news/documents."""
    store = state.get("store")
    if store is None:
        return {"retrieved_context": []}
    query = f"What's driving {state['ticker']} recently?"
    return {"retrieved_context": retrieve_with_retry(query, store)}


def technical_node(state: AgentState) -> dict:
    """
    Stage 3a: reasons over price_summary + technical_indicators only — no
    news, no market context. Runs in parallel with intel_node, quant_node,
    and risk_node; all four feed decision_node. Deliberately scoped to
    price data alone so its read isn't contaminated by sentiment.

    Indicators (from data/ingest_prices.py's fetch_technical_indicators):
    5/20-day MA crossover, RSI(14), MACD(12/26/9), Bollinger Band width,
    20-day volatility, and volume change vs. the 20-day average — computed
    directly from OHLCV data, not by the LLM.
    """
    indicators = state.get("technical_indicators") or {}
    if indicators:
        indicators_text = (
            f"MA5={indicators.get('ma5')}, MA20={indicators.get('ma20')}, "
            f"crossover={indicators.get('ma_crossover')}, "
            f"RSI(14)={indicators.get('rsi_14')}, "
            f"MACD={indicators.get('macd')} "
            f"(signal={indicators.get('macd_signal')}, "
            f"histogram={indicators.get('macd_histogram')}), "
            f"Bollinger Band width={indicators.get('bollinger_band_width_pct')}%, "
            f"20-day volatility={indicators.get('volatility_20d_pct')}%, "
            f"volume change vs. 20-day avg={indicators.get('volume_change_pct')}%"
        )
    else:
        indicators_text = "unavailable (insufficient price history)"

    messages = [
        SystemMessage(
            "You are a technical analyst. Judge only the price action and "
            "indicators given — trend and momentum (moving-average "
            "crossover, MACD), overbought/oversold signal (RSI), "
            "volatility/risk context (Bollinger Band width, 20-day "
            "volatility), and whether volume confirms the move. Do not "
            "reference news or sentiment; that's a separate analyst's job."
        ),
        HumanMessage(
            f"Ticker: {state['ticker']}\n"
            f"Price summary: {state.get('price_summary', 'unavailable')}\n"
            f"Technical indicators: {indicators_text}\n\n"
            "Give a short technical read."
        ),
    ]
    response = _llm.invoke(messages)
    return {"technical_analysis": response.content}


def intel_node(state: AgentState) -> dict:
    """
    Stage 3b: reasons over retrieved_context (corrective-RAG news) only —
    no price data. Runs in parallel with technical_node, quant_node, and
    risk_node; all four feed decision_node.

    v1: single-ticker news only, via retrieve_and_grade_node's output.
    Market-wide risk (SPY/QQQ price + news) is the planned next pass.
    """
    context = "\n".join(f"- {c['content']}" for c in state.get("retrieved_context", []))
    messages = [
        SystemMessage(
            "You are a news/sentiment analyst. Judge only the retrieved "
            "headlines given — what's driving sentiment, and any risk they "
            "imply. Do not reference price data; that's a separate "
            "analyst's job."
        ),
        HumanMessage(
            f"Ticker: {state['ticker']}\n"
            f"Recent news context:\n{context or 'none retrieved'}\n\n"
            "Give a short read on sentiment and risk."
        ),
    ]
    response = _llm.invoke(messages)
    return {"intel_analysis": response.content}


def quant_node(state: AgentState) -> dict:
    """
    Stage 3c: runs the trained LightGBM classifier — technical indicators
    plus an AutoARIMA forecast as features (see ml/features.py) — to get
    a probability that the 5-day forward return is positive. Runs in
    parallel with technical_node, intel_node, and risk_node; it's an
    independent quantitative signal that doesn't depend on any of their
    outputs.

    This is a genuinely different kind of signal from technical_node:
    technical_node is an LLM *reasoning in words* over indicator values;
    quant_node is a trained classifier's *calibrated probability* from
    the same underlying data plus a classical statistical forecast.
    Returns quant_signal=None (not an error) if ml/train.py hasn't been
    run yet — downstream nodes treat a missing quant signal as "no
    opinion."
    """
    return {"quant_signal": predict_5d_up_probability(state["ticker"])}


def risk_node(state: AgentState) -> dict:
    """
    RiskAgent — dedicated risk screening specialist.

    Responsible for:
    - Scanning for insider sell-downs, earnings warnings, regulatory actions
    - Checking valuation anomalies (PE/PB extremes)
    - Evaluating lock-up expiration risks
    - Producing risk flags that can override or downgrade signals from other agents

    Risk flags use a two-level severity system:
    - soft: downgrades the signal and adds a visible warning
    - hard: vetoes buy signals entirely when risk override is enabled

    Implementation notes: runs in parallel with technical_node, intel_node,
    and quant_node — all four fan out from retrieve_and_grade_node and fan
    into decision_node. Risk detection here is self-contained (its own
    targeted RAG query + a fundamentals fetch), so it doesn't need the
    other three agents' reads to do its job — it flags what it flags
    regardless of what technical/intel/quant concluded. The "override
    other agents' signals" part happens downstream, in decision_node's
    apply_risk_override(), which acts on the *final* combined decision
    rather than on risk_node needing to see the others' intermediate
    reads first. Insider/earnings/regulatory/lock-up signals come from a
    targeted corrective-RAG query against the same news store
    retrieve_and_grade_node already built — no new data source. Valuation
    anomalies come from a real fundamentals fetch.
    """
    store = state.get("store")
    risk_context = []
    if store is not None:
        risk_query = (
            f"{state['ticker']} insider selling, earnings warning, "
            "regulatory investigation, or lock-up expiration"
        )
        risk_context = retrieve_with_retry(risk_query, store)
    news_text = "\n".join(f"- {c['content']}" for c in risk_context) or "none found"

    fundamentals = fetch_fundamentals(state["ticker"])

    messages = [
        SystemMessage(
            "You are a risk screening specialist. Flag concrete risks only "
            "— insider sell-downs, earnings warnings, regulatory actions, "
            "lock-up expirations, or valuation extremes (PE/PB far outside "
            "a normal range). Don't invent risks the data doesn't support. "
            "severity='hard' only for something that should block a buy "
            "outright; 'soft' for something worth a visible warning but "
            "not a veto. Empty flags list if nothing concrete is found."
        ),
        HumanMessage(
            f"Ticker: {state['ticker']}\n"
            f"Risk-focused news search:\n{news_text}\n"
            f"Fundamentals: trailing P/E={fundamentals.get('trailing_pe')}, "
            f"forward P/E={fundamentals.get('forward_pe')}, "
            f"price/book={fundamentals.get('price_to_book')}\n\n"
            "List any risk flags found."
        ),
    ]
    result: RiskAssessment = _llm.with_structured_output(RiskAssessment).invoke(messages)
    return {"risk_assessment": result.model_dump()}


def compute_risk_level(flags: list[dict]) -> Literal["low", "medium", "high"]:
    """
    Deterministic risk badge from risk_node's flags — "high" if any hard
    flag is present, "medium" if only soft flags, "low" otherwise. This is
    a plain lookup, not an LLM judgment call, since RiskFlag.severity
    already encodes exactly this.
    """
    severities = {f["severity"] for f in flags}
    if "hard" in severities:
        return "high"
    if "soft" in severities:
        return "medium"
    return "low"


def apply_risk_override(result: DecisionOutput, flags: list[dict]) -> DecisionOutput:
    """
    Enforces RiskAgent's two-level severity contract on decision_node's
    output. Applied after the LLM call, not inside the prompt, so the
    override is guaranteed rather than something the model might forget:
    - hard flag + RISK_OVERRIDE_ENABLED: forces the decision down to at
      most "sell" — a hard risk finding (e.g. an active regulatory
      investigation) should read as "high risk, sell," not a neutral
      "hold." Already-bearish decisions ("sell"/"strong_sell") pass
      through unchanged; there's nothing further to veto.
    - soft flag (no hard, or override disabled): downgrades one level on
      DECISION_LEVELS (e.g. "strong_buy" -> "buy", "buy" -> "hold").
    """
    if not flags:
        return result

    hard_flags = [f for f in flags if f["severity"] == "hard"]
    soft_flags = [f for f in flags if f["severity"] == "soft"]
    current_idx = DECISION_LEVELS.index(result.decision)
    sell_idx = DECISION_LEVELS.index("sell")

    if hard_flags and RISK_OVERRIDE_ENABLED:
        if current_idx <= sell_idx:
            return result  # already sell/strong_sell — nothing to override
        warning = "; ".join(f["detail"] for f in hard_flags)
        return result.model_copy(
            update={
                "decision": "sell",
                "rationale": f"{result.rationale}\n[RISK OVERRIDE — hard flag forced a sell: {warning}]",
            }
        )
    if soft_flags:
        if current_idx == 0:
            return result  # already the most bearish level
        warning = "; ".join(f["detail"] for f in soft_flags)
        return result.model_copy(
            update={
                "decision": DECISION_LEVELS[current_idx - 1],
                "rationale": f"{result.rationale}\n[RISK WARNING — downgraded: {warning}]",
            }
        )
    return result


def decision_node(state: AgentState) -> dict:
    """
    Stage 4: combines technical_node + intel_node + quant_node + risk_node
    into one structured decision — a 5-level rating (strong_buy ...
    strong_sell), an optional buy_range and sell_range, and a rationale —
    via structured output, plus a human-readable `analysis` string so
    draft_summary_node keeps working unchanged. If technical and intel
    disagree, the rationale is told to say so explicitly rather than
    silently picking a side; risk flags are enforced afterward by
    apply_risk_override(), not left to the LLM's discretion. `risk_level`
    (low/medium/high) is a deterministic lookup from the risk flags, not
    part of the LLM's output — see compute_risk_level().
    """
    technical = state.get("technical_analysis", "none")
    intel = state.get("intel_analysis", "none")
    risk = state.get("risk_assessment") or {"flags": [], "summary": "none"}
    quant = state.get("quant_signal")
    quant_text = (
        f"{quant['probability_up']:.0%} probability of a positive 5-day return"
        if quant
        else "unavailable (model not trained — python -m ml.train)"
    )
    messages = [
        SystemMessage(
            "You combine a technical read, a news/sentiment read, and a "
            "quant model's probability estimate into one final call: a "
            "5-level rating (strong_buy, buy, hold, sell, strong_sell), "
            "plus a buy_range (price band worth considering as an entry) "
            "and a sell_range (price band worth considering as an exit) "
            "if the data supports specific levels — omit either range "
            "rather than guessing if it doesn't. If the technical and "
            "intel reads disagree, say so explicitly in the rationale "
            "instead of silently picking a side. Factor in the risk "
            "assessment given, but you don't need to enforce its severity "
            "rules yourself — that's handled separately."
        ),
        HumanMessage(
            f"Ticker: {state['ticker']}\n"
            f"Technical analysis: {technical}\n"
            f"News/intel analysis: {intel}\n"
            f"Quant model signal: {quant_text}\n"
            f"Risk assessment: {risk['summary']}\n\n"
            "Give a final rating with a rationale."
        ),
    ]
    result: DecisionOutput = _llm.with_structured_output(DecisionOutput).invoke(messages)
    result = apply_risk_override(result, risk["flags"])
    risk_level = compute_risk_level(risk["flags"])

    def format_range(r: Optional[PriceRange]) -> str:
        return f"${r.low:.2f}-${r.high:.2f}" if r else "n/a"

    analysis_text = (
        f"Rating: {result.decision.upper().replace('_', ' ')} (risk: {risk_level})"
        f"\nBuy range: {format_range(result.buy_range)}"
        f"\nSell range: {format_range(result.sell_range)}"
        f"\nRationale: {result.rationale}"
        + f"\n\nTechnical: {technical}\nIntel: {intel}\nQuant: {quant_text}\nRisk: {risk['summary']}"
    )
    decision_dict = result.model_dump()
    decision_dict["risk_level"] = risk_level
    return {"decision": decision_dict, "analysis": analysis_text}


def draft_summary_node(state: AgentState) -> dict:
    """
    Stage 5: produces the actual draft text, using the analysis notes.

    Explicitly told to state decision_node's rating, risk level, and
    buy/sell ranges rather than soften or omit them — those were
    computed for a reason and should actually reach the reader, with a
    disclaimer alongside them rather than instead of them.
    """
    prompt = (
        f"Ticker: {state['ticker']}\n"
        f"Price summary: {state.get('price_summary', 'unavailable')}\n"
        f"Analysis notes: {state.get('analysis', 'none')}\n\n"
        "Write a short (4-6 sentence) daily summary: what happened, why, "
        "and what to watch next. State the rating, risk level, and "
        "buy/sell ranges from the analysis notes explicitly and clearly — "
        "don't soften, hedge, or omit them; if a range wasn't given, say "
        "so rather than inventing one. End with 'This is informational "
        "only, not financial advice.' on its own line, followed by "
        "exactly: 'Status: draft — pending human review.'"
    )
    response = _llm.invoke(prompt)
    return {"draft": response.content}


def human_checkpoint_node(state: AgentState) -> dict:
    """
    Stage 6: for now, a CLI prompt is fine. Show the draft, ask for
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
    """Stage 7: logs the outcome and emails the draft if it was approved."""
    print(f"Logged. Approved: {state.get('approved')}")
    if state.get("approved"):
        send_notification_email(state["ticker"], state.get("draft", ""))
    return {}


def build_graph():
    """Assembles and compiles the StateGraph. Call once, reuse the result."""
    graph = StateGraph(AgentState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("retrieve_and_grade", retrieve_and_grade_node)
    graph.add_node("technical", technical_node)
    graph.add_node("intel", intel_node)
    graph.add_node("quant", quant_node)
    graph.add_node("risk", risk_node)
    graph.add_node("decision", decision_node)
    graph.add_node("draft_summary", draft_summary_node)
    graph.add_node("human_checkpoint", human_checkpoint_node)
    graph.add_node("log_and_notify", log_and_notify_node)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "retrieve_and_grade")
    # Fan-out: technical, intel, quant, and risk all run off
    # retrieve_and_grade's output, in the same superstep (LangGraph runs
    # them concurrently) — none of the four depends on any of the others'
    # output. Fan-in: decision only runs once all four have completed.
    graph.add_edge("retrieve_and_grade", "technical")
    graph.add_edge("retrieve_and_grade", "intel")
    graph.add_edge("retrieve_and_grade", "quant")
    graph.add_edge("retrieve_and_grade", "risk")
    graph.add_edge("technical", "decision")
    graph.add_edge("intel", "decision")
    graph.add_edge("quant", "decision")
    graph.add_edge("risk", "decision")
    graph.add_edge("decision", "draft_summary")
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
