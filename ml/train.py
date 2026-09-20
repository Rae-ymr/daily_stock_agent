"""
Builds a pooled training panel across STOCK_LIST tickers — technical
indicators + an AutoARIMA forecast as features, "is the 5-day forward
return positive?" as the label — trains a LightGBM binary classifier,
and saves it to ml/model.pkl.

Uses a time-based train/test split (TRAIN_TEST_SPLIT_DATE), not a random
shuffle. Shuffling would let rows *after* the split date train the model
that then gets evaluated on rows *before* it — a direct form of the
leakage / train-serve-skew problem: a time-series model must only ever
be evaluated on data strictly later in time than everything it trained
on, mirroring how it will actually be used (predicting the future from
the past, never the reverse).

This fits an AutoARIMA model once per sampled historical row (an order
search, not a lookup). Pooling multiple tickers over years of history
means hundreds of fits; expect a few minutes, not seconds, though this
is meaningfully cheaper than a foundation-model forward pass would be.

Run standalone:
    python -m ml.train
"""

import os
from typing import Optional

import joblib
import lightgbm as lgb
import pandas as pd
import yfinance as yf
from sklearn.metrics import accuracy_score, roc_auc_score

from ml.features import FEATURE_COLUMNS, build_feature_row

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
LOOKBACK_PERIOD = "5y"
FORWARD_DAYS = 5
SAMPLE_STRIDE_DAYS = 5  # skip most days — adjacent 5-day-forward labels overlap heavily
TRAIN_TEST_SPLIT_DATE = "2024-01-01"  # before this = train, on/after = test


def _label_forward_return(close: pd.Series, idx: int) -> Optional[int]:
    """1 if close FORWARD_DAYS after `idx` is higher than at `idx`, else 0.
    None near the end of the series, where there isn't FORWARD_DAYS of
    future data yet to compute a label from."""
    if idx + FORWARD_DAYS >= len(close):
        return None
    return int(close.iloc[idx + FORWARD_DAYS] > close.iloc[idx])


def build_training_panel(tickers: list) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        history = yf.Ticker(ticker).history(period=LOOKBACK_PERIOD)
        if history.empty:
            print(f"  {ticker}: no price history, skipping")
            continue

        close = history["Close"]
        ticker_rows = 0
        for idx in range(30, len(history) - FORWARD_DAYS, SAMPLE_STRIDE_DAYS):
            label = _label_forward_return(close, idx)
            features = build_feature_row(history, idx)
            if label is None or features is None:
                continue
            rows.append({**features, "label": label, "ticker": ticker, "date": history.index[idx]})
            ticker_rows += 1
        print(f"  {ticker}: {ticker_rows} usable rows")

    return pd.DataFrame(rows)


def train() -> None:
    tickers = [t.strip() for t in os.getenv("STOCK_LIST", "AAPL,MSFT").split(",") if t.strip()]
    print(f"Building training panel for {tickers}...")
    panel = build_training_panel(tickers)
    print(f"Total: {len(panel)} rows across {len(tickers)} tickers")
    if panel.empty:
        print("No usable rows — nothing to train on.")
        return

    train_df = panel[panel["date"] < TRAIN_TEST_SPLIT_DATE]
    test_df = panel[panel["date"] >= TRAIN_TEST_SPLIT_DATE]
    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows (time-based split at {TRAIN_TEST_SPLIT_DATE})")
    if train_df.empty:
        print("No rows before the split date — nothing to train on.")
        return

    model = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05)
    model.fit(train_df[FEATURE_COLUMNS], train_df["label"])

    if not test_df.empty:
        preds = model.predict(test_df[FEATURE_COLUMNS])
        probs = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
        print(f"Test accuracy: {accuracy_score(test_df['label'], preds):.3f}")
        if test_df["label"].nunique() > 1:
            print(f"Test AUC: {roc_auc_score(test_df['label'], probs):.3f}")
        else:
            print("Test AUC: undefined (test set has only one label class)")
    else:
        print("No test rows past the split date — can't report held-out metrics.")

    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train()
