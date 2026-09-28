"""
Single-ticker historical contextual-bandit environment.

Each episode presents features available at one historical decision date.
The policy chooses one of the agent's five rating levels, interpreted as a
target long-only allocation. The reward is measured over the next five
trading days and the episode then terminates.

Run a smoke test with downloaded data:
    python -m rl.environment AAPL
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from ml.features import FEATURE_COLUMNS, build_feature_row

ACTION_LABELS = (
    "strong_sell",
    "sell",
    "hold",
    "buy",
    "strong_buy",
)

# Ratings represent target long-only allocations. This keeps "sell" actions
# meaningfully more defensive than "hold" without treating them as short sales.
ACTION_ALLOCATIONS = (0.0, 0.25, 0.5, 0.75, 1.0)

FeatureBuilder = Callable[[pd.DataFrame, int], Optional[dict]]


@dataclass(frozen=True)
class BanditEpisode:
    """Precomputed context and future outcome for one decision date."""

    as_of_idx: int
    date: object
    observation: np.ndarray
    entry_price: float
    exit_price: float
    future_return: float
    future_volatility: float


class HistoricalBanditEnvironment:
    """
    Offline one-step environment for learning a rating from historical data.

    Reward values are decimal returns, not percentages:

        allocation * future_return
        - allocation * transaction_cost_bps / 10_000
        - risk_aversion * allocation**2 * future_variance

    The observation is ordered exactly as ``ml.features.FEATURE_COLUMNS``.
    Features are built only from rows through the decision date; prices after
    that date are used solely to calculate reward.
    """

    def __init__(
        self,
        price_history: pd.DataFrame,
        *,
        horizon_days: int = 5,
        sample_stride_days: int = 5,
        min_history_days: int = 30,
        transaction_cost_bps: float = 10.0,
        risk_aversion: float = 0.0,
        feature_builder: FeatureBuilder = build_feature_row,
    ) -> None:
        self.horizon_days = self._positive_int(horizon_days, "horizon_days")
        self.sample_stride_days = self._positive_int(
            sample_stride_days, "sample_stride_days"
        )
        self.min_history_days = self._positive_int(
            min_history_days, "min_history_days"
        )
        if transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be non-negative")
        if risk_aversion < 0:
            raise ValueError("risk_aversion must be non-negative")

        self.transaction_cost_bps = float(transaction_cost_bps)
        self.risk_aversion = float(risk_aversion)
        self._feature_builder = feature_builder
        self._history = self._validate_history(price_history)
        self._episodes = self._build_episodes()
        if not self._episodes:
            raise ValueError(
                "No usable episodes were produced; provide more history or "
                "check that the feature builder returns complete finite features"
            )

        self._rng = np.random.default_rng()
        self._current_episode: Optional[BanditEpisode] = None
        self._episode_done = True

    @classmethod
    def from_ticker(
        cls,
        ticker: str,
        *,
        period: str = "5y",
        **kwargs,
    ) -> "HistoricalBanditEnvironment":
        """Downloads OHLCV history with yfinance and builds an environment."""
        history = yf.Ticker(ticker).history(period=period)
        if history.empty:
            raise ValueError(f"No price history found for {ticker}")
        return cls(history, **kwargs)

    @staticmethod
    def _positive_int(value: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _validate_history(price_history: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(price_history, pd.DataFrame):
            raise TypeError("price_history must be a pandas DataFrame")
        required = {"Close", "Volume"}
        missing = required.difference(price_history.columns)
        if missing:
            raise ValueError(f"price_history is missing columns: {sorted(missing)}")
        if price_history.index.has_duplicates:
            raise ValueError("price_history index must not contain duplicate dates")

        history = price_history.sort_index().copy()
        if history["Close"].isna().any() or (history["Close"] <= 0).any():
            raise ValueError("Close prices must be positive and non-null")
        if history["Volume"].isna().any() or (history["Volume"] < 0).any():
            raise ValueError("Volume must be non-negative and non-null")
        return history

    def _build_episodes(self) -> list[BanditEpisode]:
        episodes = []
        stop = len(self._history) - self.horizon_days
        for as_of_idx in range(
            self.min_history_days - 1, stop, self.sample_stride_days
        ):
            # Pass a physically truncated frame as an additional leakage guard;
            # even a custom feature builder cannot inspect future reward rows.
            history_as_of = self._history.iloc[: as_of_idx + 1]
            features = self._feature_builder(history_as_of, as_of_idx)
            if features is None:
                continue

            missing = [name for name in FEATURE_COLUMNS if name not in features]
            if missing:
                raise ValueError(f"feature builder omitted columns: {missing}")
            observation = np.asarray(
                [features[name] for name in FEATURE_COLUMNS], dtype=np.float64
            )
            if not np.isfinite(observation).all():
                continue

            entry_price = float(self._history["Close"].iloc[as_of_idx])
            exit_idx = as_of_idx + self.horizon_days
            exit_price = float(self._history["Close"].iloc[exit_idx])
            future_return = exit_price / entry_price - 1.0

            future_prices = self._history["Close"].iloc[as_of_idx : exit_idx + 1]
            future_daily_returns = future_prices.pct_change().dropna()
            future_volatility = (
                float(future_daily_returns.std(ddof=0))
                if len(future_daily_returns)
                else 0.0
            )

            episodes.append(
                BanditEpisode(
                    as_of_idx=as_of_idx,
                    date=self._history.index[as_of_idx],
                    observation=observation,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    future_return=float(future_return),
                    future_volatility=future_volatility,
                )
            )
        return episodes

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def observation_size(self) -> int:
        return len(FEATURE_COLUMNS)

    def iter_training_samples(self):
        """Yields leakage-safe contexts and realized outcomes in date order."""
        for episode in self._episodes:
            yield {
                "date": episode.date,
                "observation": episode.observation.copy(),
                "future_return": episode.future_return,
                "future_volatility": episode.future_volatility,
            }

    def sample_action(self) -> int:
        """Samples an integer action in the inclusive range 0..4."""
        return int(self._rng.integers(len(ACTION_LABELS)))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Selects an episode and returns ``(observation, info)``.

        Pass ``options={"episode_index": n}`` for deterministic walk-forward
        evaluation. Otherwise an episode is sampled using the seeded RNG.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        options = options or {}
        episode_index = options.get("episode_index")
        if episode_index is None:
            episode_index = int(self._rng.integers(len(self._episodes)))
        if (
            isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or not 0 <= episode_index < len(self._episodes)
        ):
            raise IndexError(
                f"episode_index must be between 0 and {len(self._episodes) - 1}"
            )

        self._current_episode = self._episodes[episode_index]
        self._episode_done = False
        return self._current_episode.observation.copy(), {
            "episode_index": episode_index,
            "date": self._current_episode.date,
            "entry_price": self._current_episode.entry_price,
        }

    def step(self, action: int | str) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Applies one rating and returns a Gym-like five-tuple.

        Returns ``(observation, reward, terminated, truncated, info)``. Because
        this is a contextual bandit, every valid step has ``terminated=True``
        and ``truncated=False``.
        """
        if self._current_episode is None or self._episode_done:
            raise RuntimeError("Call reset() before step(); each episode has one step")

        action_index = self._action_index(action)
        action_label = ACTION_LABELS[action_index]
        allocation = ACTION_ALLOCATIONS[action_index]
        episode = self._current_episode

        gross_reward = allocation * episode.future_return
        transaction_cost = allocation * self.transaction_cost_bps / 10_000.0
        future_variance = episode.future_volatility**2
        risk_penalty = self.risk_aversion * allocation**2 * future_variance
        reward = gross_reward - transaction_cost - risk_penalty

        self._episode_done = True
        info = {
            "date": episode.date,
            "action": action_label,
            "allocation": allocation,
            "entry_price": episode.entry_price,
            "exit_price": episode.exit_price,
            "future_return": episode.future_return,
            "future_volatility": episode.future_volatility,
            "future_variance": future_variance,
            "gross_reward": gross_reward,
            "transaction_cost": transaction_cost,
            "risk_penalty": risk_penalty,
        }
        return episode.observation.copy(), float(reward), True, False, info

    @staticmethod
    def _action_index(action: int | str) -> int:
        if isinstance(action, str):
            try:
                return ACTION_LABELS.index(action)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown action {action!r}; expected one of {ACTION_LABELS}"
                ) from exc
        if (
            isinstance(action, bool)
            or not isinstance(action, (int, np.integer))
            or not 0 <= int(action) < len(ACTION_LABELS)
        ):
            raise ValueError(
                f"action must be an integer 0..{len(ACTION_LABELS) - 1} "
                f"or one of {ACTION_LABELS}"
            )
        return int(action)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect one historical contextual-bandit episode"
    )
    parser.add_argument("ticker", nargs="?", default="AAPL")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--episode-index", type=int, default=0)
    args = parser.parse_args()

    env = HistoricalBanditEnvironment.from_ticker(args.ticker, period=args.period)
    observation, reset_info = env.reset(
        options={"episode_index": args.episode_index}
    )
    print(
        f"{args.ticker}: {len(env)} episodes, "
        f"{env.observation_size} features per observation"
    )
    print(f"Episode: {reset_info}")
    print(f"Observation: {dict(zip(FEATURE_COLUMNS, observation))}")

    # Reset before every action because each bandit episode is one step.
    for action in ACTION_LABELS:
        env.reset(options={"episode_index": args.episode_index})
        _, reward, _, _, info = env.step(action)
        print(
            f"{action:>11}: allocation={info['allocation']:.0%}, "
            f"reward={reward:+.4%}"
        )


if __name__ == "__main__":
    main()
