"""Record, settle, and learn from delayed five-trading-day feedback."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from rl.ledger import PredictionLedger
from rl.policy import (
    DEFAULT_POLICY_PATH,
    OnlineReturnPolicy,
    clear_policy_cache,
)

HistoryFetcher = Callable[[str, date, date], pd.DataFrame]


def _fetch_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    return yf.Ticker(ticker).history(start=start.isoformat(), end=end.isoformat())


def record_live_prediction(
    ticker: str,
    quant_signal: dict,
    policy_signal: dict,
    *,
    ledger: Optional[PredictionLedger] = None,
) -> Optional[str]:
    """Records the latest market close and the policy's shadow prediction."""
    features = quant_signal.get("features") if quant_signal else None
    if not features or not policy_signal:
        return None
    history = yf.Ticker(ticker).history(period="5d")
    if history.empty:
        return None
    latest = history.iloc[-1]
    prediction_date = history.index[-1].date().isoformat()
    ledger = ledger or PredictionLedger()
    return ledger.record(
        ticker=ticker,
        prediction_date=prediction_date,
        entry_price=float(latest["Close"]),
        features=features,
        signal=policy_signal,
        horizon_days=5,
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
        exit_price = float(horizon["Close"].iloc[-1])
        entry_price = float(prediction["entry_price"])
        forward_return = exit_price / entry_price - 1.0
        prices = pd.Series(
            [entry_price, *horizon["Close"].astype(float).tolist()],
            dtype=float,
        )
        future_volatility = float(prices.pct_change().dropna().std(ddof=0))
        allocation = float(prediction["allocation"])
        transaction_cost = allocation * transaction_cost_bps / 10_000.0
        risk_penalty = (
            risk_aversion * allocation**2 * future_volatility**2
        )
        reward = allocation * forward_return - transaction_cost - risk_penalty
        settled_date = horizon.index[-1].date().isoformat()

        ledger.settle(
            prediction["id"],
            settled_date=settled_date,
            exit_price=exit_price,
            forward_return=forward_return,
            future_volatility=future_volatility,
            reward=reward,
        )
        settled_ids.append(prediction["id"])
    return settled_ids


def update_policy_from_settled(
    *,
    ledger: Optional[PredictionLedger] = None,
    policy_path=DEFAULT_POLICY_PATH,
) -> int:
    """Applies all untrained settled labels, saves, then marks them consumed."""
    ledger = ledger or PredictionLedger()
    rows = ledger.untrained_settled()
    if not rows:
        return 0

    policy = OnlineReturnPolicy.load(policy_path)
    if policy is None:
        raise RuntimeError(
            "No bootstrap policy found; run `python -m rl.train_policy` first"
        )
    observations = [
        OnlineReturnPolicy.observation_from_features(row["features"])
        for row in rows
    ]
    returns = [float(row["forward_return"]) for row in rows]
    policy.update(observations, returns)
    policy.save(policy_path)
    ledger.mark_trained([row["id"] for row in rows])
    clear_policy_cache()
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process delayed policy feedback")
    parser.add_argument(
        "command",
        choices=("settle", "update", "all"),
        default="all",
        nargs="?",
    )
    args = parser.parse_args()

    if args.command in {"settle", "all"}:
        settled = settle_pending_predictions()
        print(f"Settled {len(settled)} prediction(s)")
    if args.command in {"update", "all"}:
        updated = update_policy_from_settled()
        print(f"Updated policy with {updated} prediction(s)")


if __name__ == "__main__":
    main()
