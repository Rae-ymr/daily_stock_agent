"""
5-day return forecast via Nixtla's AutoARIMA — a classical statistical
forecaster, not a foundation model. Used purely as one input feature to
the LightGBM classifier in ml/train.py and ml/predict.py.

AutoARIMA searches over (p, d, q) orders and picks the best fit by AIC
for each series it's given — there's no pretrained weights to download
and no GPU involved, unlike a foundation-model forecaster. Much cheaper
per call, at the cost of only capturing linear autoregressive patterns
rather than whatever a large pretrained model might pick up.

Run standalone:
    python -m ml.autoarima_forecast AAPL
"""

import sys
from typing import Optional

import pandas as pd
from statsforecast.models import AutoARIMA

FORECAST_HORIZON_DAYS = 5
MIN_CONTEXT_LENGTH = 30


def forecast_5d_return(close_prices: pd.Series) -> Optional[float]:
    """
    Fits AutoARIMA on the trailing close-price series and returns the
    forecasted % return FORECAST_HORIZON_DAYS trading days out (point
    forecast vs. the last known close). None if there's too little
    history, or AutoARIMA fails to fit (e.g. a degenerate/constant
    series) — this is a best-effort feature, not a critical path, so a
    fitting failure should fall back to "no signal," not propagate.
    """
    if len(close_prices) < MIN_CONTEXT_LENGTH:
        return None

    try:
        model = AutoARIMA()
        model.fit(y=close_prices.values.astype(float))
        forecast = model.predict(h=FORECAST_HORIZON_DAYS)
        forecasted_price = float(forecast["mean"][-1])
    except Exception:
        return None

    last_close = float(close_prices.iloc[-1])
    return float((forecasted_price - last_close) / last_close * 100)


if __name__ == "__main__":
    import yfinance as yf

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    history = yf.Ticker(ticker).history(period="3mo")
    result = forecast_5d_return(history["Close"])
    print(f"{ticker}: AutoARIMA forecasted {FORECAST_HORIZON_DAYS}-day return = {result}%")
