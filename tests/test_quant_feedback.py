import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ml.features import FEATURE_COLUMNS
from rl.feedback import retrain_quant_from_settled, settle_pending_predictions
from rl.ledger import PredictionLedger


def features() -> dict:
    return {
        name: float(index + 1)
        for index, name in enumerate(FEATURE_COLUMNS)
    }


class QuantFeedbackTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.ledger = PredictionLedger(
            Path(self.temporary_directory.name) / "predictions.db"
        )

    def record_prediction(self) -> str:
        return self.ledger.record(
            ticker="MSFT",
            prediction_date="2024-01-02",
            entry_price=100.0,
            features=features(),
            probability_up=0.7,
            decision="buy",
            allocation=0.75,
        )

    def test_record_is_idempotent_and_settles_after_five_sessions(self):
        prediction_id = self.record_prediction()
        self.assertEqual(prediction_id, self.record_prediction())
        future_history = pd.DataFrame(
            {"Close": [101.0, 102.0, 103.0, 104.0, 105.0]},
            index=pd.date_range("2024-01-03", periods=5, freq="B"),
        )

        settled = settle_pending_predictions(
            ledger=self.ledger,
            history_fetcher=lambda _ticker, _start, _end: future_history,
            today=date(2024, 1, 10),
        )

        self.assertEqual(settled, [prediction_id])
        row = self.ledger.get(prediction_id)
        self.assertEqual(row["status"], "settled")
        self.assertEqual(row["label_up"], 1)
        self.assertEqual(row["direction_correct"], 1)
        self.assertAlmostEqual(row["forward_return"], 0.05)
        self.assertGreater(row["reward"], 0)

    def test_settled_label_is_retrained_and_consumed_once(self):
        prediction_id = self.record_prediction()
        self.ledger.settle(
            prediction_id,
            settled_date="2024-01-09",
            exit_price=97.0,
            forward_return=-0.03,
            label_up=0,
            direction_correct=False,
            future_volatility=0.01,
            reward=-0.02,
        )
        captured_rows = []

        def successful_train(extra_rows):
            captured_rows.extend(extra_rows)
            return True

        with patch("rl.feedback.train", side_effect=successful_train):
            trained = retrain_quant_from_settled(
                ledger=self.ledger,
                min_labels=1,
            )
            second_training = retrain_quant_from_settled(
                ledger=self.ledger,
                min_labels=1,
            )

        self.assertEqual(trained, 1)
        self.assertEqual(second_training, 0)
        self.assertEqual(captured_rows[0]["label"], 0)
        self.assertEqual(captured_rows[0]["ticker"], "MSFT")
        self.assertIsNotNone(self.ledger.get(prediction_id)["trained_at"])

    def test_retraining_waits_for_minimum_batch(self):
        prediction_id = self.record_prediction()
        self.ledger.settle(
            prediction_id,
            settled_date="2024-01-09",
            exit_price=103.0,
            forward_return=0.03,
            label_up=1,
            direction_correct=True,
            future_volatility=0.01,
            reward=0.02,
        )

        with patch("rl.feedback.train") as train_mock:
            trained = retrain_quant_from_settled(
                ledger=self.ledger,
                min_labels=5,
            )

        self.assertEqual(trained, 0)
        train_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
