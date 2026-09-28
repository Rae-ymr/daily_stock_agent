import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ml.features import FEATURE_COLUMNS
from rl.feedback import settle_pending_predictions, update_policy_from_settled
from rl.ledger import PredictionLedger
from rl.policy import OnlineReturnPolicy


def feature_dict(scale: float = 1.0) -> dict:
    return {
        name: float((index + 1) * scale)
        for index, name in enumerate(FEATURE_COLUMNS)
    }


def fitted_policy() -> OnlineReturnPolicy:
    rng = np.random.default_rng(7)
    observations = rng.normal(size=(40, len(FEATURE_COLUMNS)))
    returns = observations[:, 0] * 0.01 + observations[:, 2] * 0.002
    return OnlineReturnPolicy(random_state=7).fit(observations, returns)


class OnlineReturnPolicyTest(unittest.TestCase):
    def test_fit_predict_update_and_persist(self):
        policy = fitted_policy()
        initial_version = policy.model_version
        signal = policy.predict(feature_dict(0.1))

        self.assertIn(
            signal.action,
            ("strong_sell", "sell", "hold", "buy", "strong_buy"),
        )
        self.assertEqual(len(signal.expected_utilities), 5)

        observation = OnlineReturnPolicy.observation_from_features(
            feature_dict(0.2)
        )
        policy.update([observation], [0.03])
        self.assertEqual(policy.model_version, initial_version + 1)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pkl"
            policy.save(path)
            loaded = OnlineReturnPolicy.load(path)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.model_version, policy.model_version)


class DelayedFeedbackTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        directory = Path(self.temporary_directory.name)
        self.ledger = PredictionLedger(directory / "predictions.db")
        self.policy_path = directory / "policy.pkl"

    def record_prediction(self) -> str:
        signal = {
            "action": "buy",
            "allocation": 0.75,
            "predicted_5d_return": 0.02,
            "confidence": 0.5,
            "model_version": 1,
        }
        return self.ledger.record(
            ticker="AAPL",
            prediction_date="2024-01-02",
            entry_price=100.0,
            features=feature_dict(),
            signal=signal,
        )

    def test_record_is_idempotent_and_settles_after_five_sessions(self):
        prediction_id = self.record_prediction()
        duplicate_id = self.record_prediction()
        self.assertEqual(prediction_id, duplicate_id)

        index = pd.date_range("2024-01-03", periods=5, freq="B")
        future_history = pd.DataFrame(
            {"Close": [101.0, 102.0, 103.0, 104.0, 105.0]},
            index=index,
        )

        settled = settle_pending_predictions(
            ledger=self.ledger,
            history_fetcher=lambda _ticker, _start, _end: future_history,
            today=date(2024, 1, 10),
        )

        self.assertEqual(settled, [prediction_id])
        row = self.ledger.get(prediction_id)
        self.assertEqual(row["status"], "settled")
        self.assertAlmostEqual(row["forward_return"], 0.05)

    def test_settled_label_updates_bootstrap_policy_once(self):
        prediction_id = self.record_prediction()
        self.ledger.settle(
            prediction_id,
            settled_date="2024-01-09",
            exit_price=103.0,
            forward_return=0.03,
            future_volatility=0.01,
            reward=0.02,
        )
        policy = fitted_policy()
        original_version = policy.model_version
        policy.save(self.policy_path)

        updated = update_policy_from_settled(
            ledger=self.ledger,
            policy_path=self.policy_path,
        )
        second_update = update_policy_from_settled(
            ledger=self.ledger,
            policy_path=self.policy_path,
        )

        self.assertEqual(updated, 1)
        self.assertEqual(second_update, 0)
        reloaded = OnlineReturnPolicy.load(self.policy_path)
        self.assertEqual(reloaded.model_version, original_version + 1)
        self.assertIsNotNone(self.ledger.get(prediction_id)["trained_at"])


if __name__ == "__main__":
    unittest.main()
