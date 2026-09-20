"""
Shared feature-building logic for the LightGBM 5-day-return classifier.

The same build_feature_row() is called by both ml/train.py (building the
historical training panel, one row per sampled past trading day) and
agent/graph.py's quant_node (live inference, one row for "today"). Using
one function in both places — rather than one implementation for
training and a separate one for serving — is the concrete fix for
train/serve skew: if the two paths computed features even slightly
differently, the model would be scored on a different feature
distribution than it was trained on.
"""

from typing import Optional

import pandas as pd

from data.ingest_prices import compute_technical_indicators_df
from ml.autoarima_forecast import forecast_5d_return

FEATURE_COLUMNS = [
    "ma5",
    "ma20",
    "ma_spread_pct",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bollinger_band_width_pct",
    "volatility_20d_pct",
    "volume_change_pct",
    "arima_forecast_5d_return",
]


def build_feature_row(price_history: pd.DataFrame, as_of_idx: int) -> Optional[dict]:
    """
    Builds one feature row "as of" the trading day at `price_history`'s
    row `as_of_idx` (inclusive) — uses only rows up to and including
    that index, never later ones, so a training row can never see
    future information it wouldn't have had at the time. Returns None
    if there isn't enough history yet, or any indicator is still NaN at
    that point (early rows before the rolling windows fill in).
    """
    history_slice = price_history.iloc[: as_of_idx + 1]
    if len(history_slice) < 30:
        return None

    indicators = compute_technical_indicators_df(history_slice)
    latest = indicators.iloc[-1]
    if latest.isna().any():
        return None

    arima_return = forecast_5d_return(history_slice["Close"])
    if arima_return is None:
        return None

    return {
        "ma5": float(latest["ma5"]),
        "ma20": float(latest["ma20"]),
        "ma_spread_pct": float(latest["ma_spread_pct"]),
        "rsi_14": float(latest["rsi_14"]),
        "macd": float(latest["macd"]),
        "macd_signal": float(latest["macd_signal"]),
        "macd_histogram": float(latest["macd_histogram"]),
        "bollinger_band_width_pct": float(latest["bollinger_band_width_pct"]),
        "volatility_20d_pct": float(latest["volatility_20d_pct"]),
        "volume_change_pct": float(latest["volume_change_pct"]),
        "arima_forecast_5d_return": arima_return,
    }
