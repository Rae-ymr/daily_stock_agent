"""
Tools the agent can call. Each one should be a small, focused function
with a clear docstring — the docstring is what the LLM sees to decide
when to call it, so be specific.

Fill in:
  - Wire these to your real data modules (data/ingest_prices.py, etc.)
  - Add the @tool decorator from your framework of choice once you
    pick LangGraph's tool-calling convention

Run standalone (sanity check the plain functions before wrapping as tools):
    python -m agent.tools AAPL
"""

import sys
from langchain_core.tools import tool

from data.ingest_prices import fetch_price_history
from data.ingest_news import fetch_news


@tool
def get_price_summary(ticker: str) -> str:
    """Returns a short text summary of recent price movement for a stock ticker.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL"
    """
    history = fetch_price_history(ticker, period="5d")
    rows = history["rows"]
    if not rows:
        return f"No price data found for {ticker}"
    first, last = rows[0], rows[-1]
    pct_change = (last["close"] - first["close"]) / first["close"] * 100
    return f"{ticker}: {first['close']} -> {last['close']} over last 5 days ({pct_change:+.2f}%)"


@tool
def get_recent_news(ticker: str) -> list[dict]:
    """Returns recent news headlines for a stock ticker, for the retrieval step to use.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL"
    """
    return fetch_news(ticker)


def check_forecast_anomaly(ticker: str) -> str:
    """
    TODO (optional, later): plug in a forecasting/anomaly-detection model
    here (e.g. Nixtla statsforecast + PyOD) if you add that module.
    """
    raise NotImplementedError


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(get_price_summary.invoke({"ticker": ticker}))
    print(get_recent_news.invoke({"ticker": ticker}))
