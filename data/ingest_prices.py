"""
Pulls daily price history for a ticker.

Fill in:
  - Any additional fields you want (volume, moving averages, etc.)
  - Caching so you don't re-fetch on every run

Run standalone:
    python -m data.ingest_prices AAPL
"""

import sys

import pandas as pd
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


def compute_technical_indicators_df(data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the full indicator time series (one row per trading day in
    `data`) — moving averages, RSI(14), MACD(12/26/9), Bollinger Band
    width, 20-day volatility, and volume change — from OHLCV `data`.

    Deliberately factored out from fetch_technical_indicators() (which
    only needs the latest row) so ml/train.py's historical training
    panel and this module's live "as of now" lookup both compute
    indicators via the exact same math. Two separate implementations
    would risk train/serve skew — the model learning on one definition
    of "RSI" and predicting on a subtly different one.
    """
    close = data["Close"]
    volume = data["Volume"]

    ma5, ma20 = close.rolling(5).mean(), close.rolling(20).mean()

    delta = close.diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi_14 = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()

    bb_std = close.rolling(20).std()
    bollinger_band_width_pct = (4 * bb_std / ma20) * 100  # (upper - lower) / middle

    volatility_20d_pct = close.pct_change().rolling(20).std() * 100

    avg_volume_20d = volume.rolling(20).mean()
    volume_change_pct = (volume - avg_volume_20d) / avg_volume_20d * 100

    return pd.DataFrame(
        {
            "ma5": ma5,
            "ma20": ma20,
            "ma_spread_pct": (ma5 - ma20) / ma20 * 100,
            "rsi_14": rsi_14,
            "macd": macd_line,
            "macd_signal": macd_signal,
            "macd_histogram": macd_line - macd_signal,
            "bollinger_band_width_pct": bollinger_band_width_pct,
            "volatility_20d_pct": volatility_20d_pct,
            "volume_change_pct": volume_change_pct,
        }
    )


def fetch_technical_indicators(ticker: str, period: str = "3mo") -> dict:
    """
    Live "as of now" indicator lookup for a ticker, for agent/graph.py's
    technical_node. Adds a human-readable moving-average crossover label
    on top of compute_technical_indicators_df()'s raw numbers.

    Needs enough history for a 26/9 MACD and a 20-day MA/Bollinger to be
    meaningful — the default period="3mo" (~60 trading days) covers that
    with room to spare. Returns {} if there's too little history (e.g. a
    very recently listed ticker).
    """
    data = yf.Ticker(ticker).history(period=period)
    if len(data) < 30:
        return {}

    indicators = compute_technical_indicators_df(data)
    latest = indicators.iloc[-1]
    ma5_prev, ma20_prev = indicators["ma5"].iloc[-2], indicators["ma20"].iloc[-2]
    ma5_now, ma20_now = latest["ma5"], latest["ma20"]

    if pd.isna(ma5_prev) or pd.isna(ma20_prev):
        crossover = None
    elif ma5_prev <= ma20_prev and ma5_now > ma20_now:
        crossover = "golden_cross"  # 5-day just crossed above 20-day — bullish
    elif ma5_prev >= ma20_prev and ma5_now < ma20_now:
        crossover = "death_cross"  # 5-day just crossed below 20-day — bearish
    else:
        crossover = "above" if ma5_now > ma20_now else "below"

    def value(key: str) -> "float | None":
        v = latest[key]
        return round(float(v), 4) if pd.notna(v) else None

    return {
        "ma5": value("ma5"),
        "ma20": value("ma20"),
        "ma_crossover": crossover,
        "rsi_14": value("rsi_14"),
        "macd": value("macd"),
        "macd_signal": value("macd_signal"),
        "macd_histogram": value("macd_histogram"),
        "bollinger_band_width_pct": value("bollinger_band_width_pct"),
        "volatility_20d_pct": value("volatility_20d_pct"),
        "volume_change_pct": value("volume_change_pct"),
    }


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    result = fetch_price_history(ticker)
    print(f"Fetched {len(result['rows'])} rows for {ticker}")
    print(result["rows"][-1] if result["rows"] else "No data")
