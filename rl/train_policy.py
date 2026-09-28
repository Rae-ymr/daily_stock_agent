"""Bootstrap the online policy from leakage-safe historical episodes."""

import argparse
import os

import numpy as np

from ml.features import FEATURE_COLUMNS
from rl.environment import HistoricalBanditEnvironment
from rl.policy import DEFAULT_POLICY_PATH, OnlineReturnPolicy


def build_dataset(tickers: list[str], period: str) -> list[dict]:
    samples = []
    for ticker in tickers:
        print(f"Building historical policy samples for {ticker}...")
        environment = HistoricalBanditEnvironment.from_ticker(
            ticker,
            period=period,
            risk_aversion=2.0,
        )
        ticker_samples = list(environment.iter_training_samples())
        for sample in ticker_samples:
            sample["ticker"] = ticker
        samples.extend(ticker_samples)
        print(f"  {len(ticker_samples)} samples")
    samples.sort(key=lambda sample: sample["date"])
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap the five-day online return policy"
    )
    parser.add_argument(
        "--tickers",
        default=os.getenv("STOCK_LIST", "AAPL,MSFT"),
        help="Comma-separated ticker symbols",
    )
    parser.add_argument("--period", default="5y")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--output", default=str(DEFAULT_POLICY_PATH))
    args = parser.parse_args()

    if not 0 < args.test_fraction < 1:
        raise ValueError("--test-fraction must be between 0 and 1")
    tickers = [ticker.strip() for ticker in args.tickers.split(",") if ticker.strip()]
    samples = build_dataset(tickers, args.period)
    if len(samples) < 10:
        raise RuntimeError("At least 10 historical samples are required")

    observations = np.asarray([sample["observation"] for sample in samples])
    returns = np.asarray([sample["future_return"] for sample in samples])
    split_idx = int(len(samples) * (1 - args.test_fraction))
    if not 1 <= split_idx < len(samples):
        raise ValueError("--test-fraction leaves no train or test samples")

    evaluation_policy = OnlineReturnPolicy().fit(
        observations[:split_idx], returns[:split_idx]
    )
    predictions = np.asarray(
        [
            evaluation_policy.predict(
                dict(zip(FEATURE_COLUMNS, observation))
            ).predicted_5d_return
            for observation in observations[split_idx:]
        ]
    )
    actual = returns[split_idx:]
    mae = float(np.mean(np.abs(predictions - actual)))
    directional_accuracy = float(np.mean((predictions > 0) == (actual > 0)))
    print(
        f"Held-out MAE: {mae:.4f}; "
        f"directional accuracy: {directional_accuracy:.1%}"
    )

    final_policy = OnlineReturnPolicy().fit(observations, returns)
    final_policy.save(args.output)
    print(
        f"Saved policy version {final_policy.model_version} with "
        f"{final_policy.samples_seen} samples to {args.output}"
    )


if __name__ == "__main__":
    main()
