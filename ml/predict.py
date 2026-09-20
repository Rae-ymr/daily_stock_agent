"""
Loads the trained LightGBM model (ml/train.py's output) and scores a
single ticker's current data. Used by agent/graph.py's quant_node.

Run standalone (needs ml/model.pkl — run `python -m ml.train` first):
    python -m ml.predict AAPL
"""

import sys
from typing import Optional

import joblib
import yfinance as yf

from ml.features import FEATURE_COLUMNS, build_feature_row
from ml.train import MODEL_PATH

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
        except FileNotFoundError:
            return None
    return _model


def predict_5d_up_probability(ticker: str) -> Optional[dict]:
    """
    Returns {"probability_up": float, "features": {...}} for `ticker`'s
    current data, or None if the model hasn't been trained yet (run
    `python -m ml.train` first) or there isn't enough price history.
    """
    model = _get_model()
    if model is None:
        return None

    history = yf.Ticker(ticker).history(period="6mo")
    if history.empty:
        return None

    features = build_feature_row(history, len(history) - 1)
    if features is None:
        return None

    row = [[features[col] for col in FEATURE_COLUMNS]]
    probability_up = float(model.predict_proba(row)[0][1])
    return {"probability_up": probability_up, "features": features}


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    result = predict_5d_up_probability(ticker)
    if result is None:
        print("No prediction — model not trained yet, or insufficient data. Run `python -m ml.train` first.")
    else:
        print(f"{ticker}: P(5-day return > 0) = {result['probability_up']:.3f}")
        print(f"Features used: {result['features']}")
