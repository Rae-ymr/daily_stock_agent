"""Independent five-day online policy trained on delayed market outcomes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from ml.features import FEATURE_COLUMNS
from rl.environment import ACTION_ALLOCATIONS, ACTION_LABELS

DEFAULT_POLICY_PATH = Path(
    os.getenv("RL_POLICY_PATH", Path(__file__).with_name("policy.pkl"))
)


@dataclass(frozen=True)
class PolicySignal:
    action: str
    allocation: float
    predicted_5d_return: float
    confidence: float
    expected_utilities: dict[str, float]
    model_version: int

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "allocation": self.allocation,
            "predicted_5d_return": self.predicted_5d_return,
            "confidence": self.confidence,
            "expected_utilities": self.expected_utilities,
            "model_version": self.model_version,
            "mode": "shadow",
        }


class OnlineReturnPolicy:
    """
    Predicts a five-day return and converts it to a long-only allocation.

    The scaler is fitted during historical bootstrap and then frozen. Delayed
    live labels update only SGDRegressor via ``partial_fit`` so old model
    coefficients remain in the same feature coordinate system.
    """

    def __init__(
        self,
        *,
        transaction_cost_bps: float = 10.0,
        risk_aversion: float = 2.0,
        random_state: int = 42,
    ) -> None:
        if transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be non-negative")
        if risk_aversion < 0:
            raise ValueError("risk_aversion must be non-negative")
        self.transaction_cost_bps = float(transaction_cost_bps)
        self.risk_aversion = float(risk_aversion)
        self.scaler = StandardScaler()
        self.model = SGDRegressor(
            loss="huber",
            penalty="l2",
            alpha=0.0001,
            learning_rate="invscaling",
            eta0=0.01,
            max_iter=2000,
            tol=1e-4,
            random_state=random_state,
        )
        self.is_fitted = False
        self.samples_seen = 0
        self.model_version = 0

    def fit(self, observations, returns) -> "OnlineReturnPolicy":
        x, y = self._training_arrays(observations, returns)
        scaled = self.scaler.fit_transform(x)
        self.model.fit(scaled, y)
        self.is_fitted = True
        self.samples_seen = len(y)
        self.model_version += 1
        return self

    def update(self, observations, returns) -> "OnlineReturnPolicy":
        """Incrementally learns from newly settled five-day labels."""
        if not self.is_fitted:
            raise RuntimeError("Bootstrap the policy with fit() before update()")
        x, y = self._training_arrays(observations, returns)
        self.model.partial_fit(self.scaler.transform(x), y)
        self.samples_seen += len(y)
        self.model_version += 1
        return self

    def predict(
        self,
        features: dict,
        *,
        volatility_20d_pct: Optional[float] = None,
    ) -> PolicySignal:
        if not self.is_fitted:
            raise RuntimeError("Policy is not fitted")
        observation = self.observation_from_features(features)
        predicted_return = float(
            self.model.predict(self.scaler.transform(observation.reshape(1, -1)))[0]
        )

        if volatility_20d_pct is None:
            volatility_20d_pct = float(features["volatility_20d_pct"])
        daily_volatility = max(float(volatility_20d_pct), 0.0) / 100.0
        five_day_variance = daily_volatility**2 * 5
        transaction_cost_rate = self.transaction_cost_bps / 10_000.0

        utilities = {}
        for label, allocation in zip(ACTION_LABELS, ACTION_ALLOCATIONS):
            utilities[label] = float(
                allocation * predicted_return
                - allocation * transaction_cost_rate
                - self.risk_aversion * allocation**2 * five_day_variance
            )
        action = max(utilities, key=utilities.get)
        action_index = ACTION_LABELS.index(action)

        five_day_volatility = daily_volatility * np.sqrt(5)
        confidence = min(
            1.0,
            abs(predicted_return) / max(five_day_volatility, 1e-6),
        )
        return PolicySignal(
            action=action,
            allocation=ACTION_ALLOCATIONS[action_index],
            predicted_5d_return=predicted_return,
            confidence=float(confidence),
            expected_utilities=utilities,
            model_version=self.model_version,
        )

    @staticmethod
    def observation_from_features(features: dict) -> np.ndarray:
        missing = [name for name in FEATURE_COLUMNS if name not in features]
        if missing:
            raise ValueError(f"Missing policy features: {missing}")
        observation = np.asarray(
            [features[name] for name in FEATURE_COLUMNS], dtype=np.float64
        )
        if not np.isfinite(observation).all():
            raise ValueError("Policy features must all be finite")
        return observation

    @staticmethod
    def _training_arrays(observations, returns) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(observations, dtype=np.float64)
        y = np.asarray(returns, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError(
                f"observations must have shape (n, {len(FEATURE_COLUMNS)})"
            )
        if y.ndim != 1 or len(y) != len(x) or not len(y):
            raise ValueError("returns must be a non-empty vector matching observations")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("training data must be finite")
        return x, y

    def save(self, path: Path = DEFAULT_POLICY_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        joblib.dump(self, temporary)
        temporary.replace(path)

    @staticmethod
    def load(path: Path = DEFAULT_POLICY_PATH) -> Optional["OnlineReturnPolicy"]:
        path = Path(path)
        if not path.exists():
            return None
        policy = joblib.load(path)
        if not isinstance(policy, OnlineReturnPolicy):
            raise TypeError(f"{path} does not contain an OnlineReturnPolicy")
        return policy


_cached_policy: Optional[OnlineReturnPolicy] = None
_cached_path: Optional[Path] = None


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> Optional[OnlineReturnPolicy]:
    """Loads and caches the live policy used by the graph's shadow node."""
    global _cached_policy, _cached_path
    path = Path(path)
    if _cached_policy is None or _cached_path != path:
        _cached_policy = OnlineReturnPolicy.load(path)
        _cached_path = path
    return _cached_policy


def clear_policy_cache() -> None:
    global _cached_policy, _cached_path
    _cached_policy = None
    _cached_path = None
