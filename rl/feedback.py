"""Record, settle, and retrain LightGBM from delayed five-day feedback."""

from __future__ import annotations

import argparse
import os
from datetime import date, timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from ml.train import train
from rl.ledger import PredictionLedger

DECISION_ALLOCATIONS = {
    "strong_sell": 0.0,
    "sell": 0.25,
    "hold": 0.5,
    "buy": 0.75,
    "strong_buy": 1.0,
}
DEFAULT_RETRAIN_MIN_LABELS = int(os.getenv("QUANT_RETRAIN_MIN_LABELS", "5"))
HistoryFetcher = Callable[[str, date, date], pd.DataFrame]


def _fetch_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    return yf.Ticker(ticker).history(start=start.isoformat(), end=end.isoformat())


def record_live_quant_prediction(
    ticker: str,
    quant_signal: dict,
    decision: dict,
    *,
    ledger: Optional[PredictionLedger] = None,
) -> Optional[str]:
    """Persists one quant prediction and the final advice shown that day."""
    if not quant_signal or not quant_signal.get("features") or not decision:
        return None
    decision_name = decision["decision"]
    if decision_name not in DECISION_ALLOCATIONS:
        raise ValueError(f"Unknown decision: {decision_name}")

    history = yf.Ticker(ticker).history(period="5d")
    if history.empty:
        return None
    ledger = ledger or PredictionLedger()
    return ledger.record(
        ticker=ticker,
        prediction_date=history.index[-1].date().isoformat(),
        entry_price=float(history["Close"].iloc[-1]),
        features=quant_signal["features"],
        probability_up=float(quant_signal["probability_up"]),
        decision=decision_name,
        allocation=DECISION_ALLOCATIONS[decision_name],
    )


def settle_pending_predictions(
    *,
    ledger: Optional[PredictionLedger] = None,
    history_fetcher: HistoryFetcher = _fetch_history,
    today: Optional[date] = None,
    transaction_cost_bps: float = 10.0,
    risk_aversion: float = 2.0,
) -> list[str]:
    """Settles predictions once five later trading sessions are available."""
    ledger = ledger or PredictionLedger()
    today = today or date.today()
    settled_ids = []

    for prediction in ledger.pending():
        prediction_date = date.fromisoformat(prediction["prediction_date"])
        history = history_fetcher(
            prediction["ticker"],
            prediction_date + timedelta(days=1),
            today + timedelta(days=2),
        )
        if history.empty:
            continue
        future_rows = history[
            np.asarray([index.date() > prediction_date for index in history.index])
        ]
        horizon_days = int(prediction["horizon_days"])
        if len(future_rows) < horizon_days:
            continue

        horizon = future_rows.iloc[:horizon_days]
        entry_price = float(prediction["entry_price"])
        exit_price = float(horizon["Close"].iloc[-1])
        forward_return = exit_price / entry_price - 1.0
        label_up = int(forward_return > 0)
        predicted_up = float(prediction["probability_up"]) >= 0.5
        prices = pd.Series(
            [entry_price, *horizon["Close"].astype(float).tolist()],
            dtype=float,
        )
        future_volatility = float(prices.pct_change().dropna().std(ddof=0))
        allocation = float(prediction["allocation"])
        reward = (
            allocation * forward_return
            - allocation * transaction_cost_bps / 10_000.0
            - risk_aversion * allocation**2 * future_volatility**2
        )

        ledger.settle(
            prediction["id"],
            settled_date=horizon.index[-1].date().isoformat(),
            exit_price=exit_price,
            forward_return=forward_return,
            label_up=label_up,
            direction_correct=predicted_up == bool(label_up),
            future_volatility=future_volatility,
            reward=reward,
        )
        settled_ids.append(prediction["id"])
    return settled_ids


def retrain_quant_from_settled(
    *,
    ledger: Optional[PredictionLedger] = None,
    min_labels: int = DEFAULT_RETRAIN_MIN_LABELS,
    force: bool = False,
) -> int:
    """Batch-retrains LightGBM and consumes settled labels only on success."""
    ledger = ledger or PredictionLedger()
    rows = ledger.untrained_settled()
    if not rows or (len(rows) < min_labels and not force):
        return 0

    supplemental = [
        {
            **row["features"],
            "label": int(row["label_up"]),
            "ticker": row["ticker"],
            "date": row["prediction_date"],
        }
        for row in rows
    ]
    if not train(extra_rows=supplemental):
        return 0
    ledger.mark_trained([row["id"] for row in rows])
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process delayed quant feedback")
    parser.add_argument(
        "command",
        choices=("settle", "retrain", "all"),
        default="all",
        nargs="?",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even below QUANT_RETRAIN_MIN_LABELS",
    )
    args = parser.parse_args()

    if args.command in {"settle", "all"}:
        settled = settle_pending_predictions()
        print(f"Settled {len(settled)} quant prediction(s)")
    if args.command in {"retrain", "all"}:
        trained = retrain_quant_from_settled(force=args.force)
        print(f"Retrained LightGBM with {trained} live-feedback label(s)")


if __name__ == "__main__":
    main()
