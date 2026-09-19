"""
Pulls daily price history for a ticker.

Fill in:
  - Any additional fields you want (volume, moving averages, etc.)
  - Caching so you don't re-fetch on every run

Run standalone:
    python -m data.ingest_prices AAPL
"""

import sys
import yfinance as yf


def fetch_price_history(ticker: str, period: str = "3mo") -> dict:
    """
    Returns basic OHLCV price history for a ticker, as a list of dicts
    (not a raw DataFrame) — JSON-serializable, and consistent with the
    dict-per-row shape ingest_news.py already returns.
    """
    data = yf.Ticker(ticker).history(period=period)
    rows = [
        {
            "date": index.strftime("%Y-%m-%d"),
            "open": round(row["Open"], 2),
            "high": round(row["High"], 2),
            "low": round(row["Low"], 2),
            "close": round(row["Close"], 2),
            "volume": int(row["Volume"]),
        }
        for index, row in data.iterrows()
    ]
    return {
        "ticker": ticker,
        "period": period,
        "rows": rows,
    }


def fetch_fundamentals(ticker: str) -> dict:
    """
    Returns basic valuation metrics for a ticker via yfinance's `.info`.
    Used by agent/graph.py's risk_node to screen for PE/PB extremes.
    Fields are None if yfinance doesn't have them for this ticker —
    common for newly-listed or thinly-covered names.
    """
    info = yf.Ticker(ticker).info
    return {
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "price_to_book": info.get("priceToBook"),
    }


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    result = fetch_price_history(ticker)
    print(f"Fetched {len(result['rows'])} rows for {ticker}")
    print(result["rows"][-1] if result["rows"] else "No data")
