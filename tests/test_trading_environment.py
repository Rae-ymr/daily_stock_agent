import unittest

import numpy as np
import pandas as pd

from ml.features import FEATURE_COLUMNS
from rl.environment import ACTION_LABELS, HistoricalBanditEnvironment


def make_history(rows: int = 12) -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=rows, freq="B")
    close = np.arange(100.0, 100.0 + rows)
    return pd.DataFrame(
        {
            "Open": close - 0.25,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": np.full(rows, 1_000_000),
        },
        index=index,
    )


def deterministic_features(_history: pd.DataFrame, as_of_idx: int) -> dict:
    return {
        name: float(as_of_idx + offset + 1)
        for offset, name in enumerate(FEATURE_COLUMNS)
    }


class HistoricalBanditEnvironmentTest(unittest.TestCase):
    def make_environment(self, **overrides) -> HistoricalBanditEnvironment:
        kwargs = {
            "horizon_days": 2,
            "sample_stride_days": 2,
            "min_history_days": 3,
            "transaction_cost_bps": 10.0,
            "risk_aversion": 0.0,
            "feature_builder": deterministic_features,
        }
        kwargs.update(overrides)
        return HistoricalBanditEnvironment(make_history(), **kwargs)

    def test_builds_expected_walk_forward_episodes(self):
        env = self.make_environment()

        self.assertEqual(len(env), 4)
        self.assertEqual(env.observation_size, len(FEATURE_COLUMNS))

        observation, info = env.reset(options={"episode_index": 0})
        self.assertEqual(observation.shape, (len(FEATURE_COLUMNS),))
        self.assertEqual(info["entry_price"], 102.0)

    def test_reward_matches_long_only_allocation_and_cost(self):
        env = self.make_environment()

        expected_return = 104.0 / 102.0 - 1.0
        expected_allocations = {
            "strong_sell": 0.0,
            "sell": 0.25,
            "hold": 0.5,
            "buy": 0.75,
            "strong_buy": 1.0,
        }

        for action, allocation in expected_allocations.items():
            with self.subTest(action=action):
                env.reset(options={"episode_index": 0})
                _, reward, terminated, truncated, info = env.step(action)
                expected_reward = allocation * expected_return - allocation * 0.001
                self.assertAlmostEqual(reward, expected_reward)
                self.assertEqual(info["allocation"], allocation)
                self.assertTrue(terminated)
                self.assertFalse(truncated)

    def test_risk_penalty_is_reported_separately(self):
        env = self.make_environment(risk_aversion=2.0)
        env.reset(options={"episode_index": 0})

        _, reward, _, _, info = env.step("strong_buy")

        expected = (
            info["future_return"]
            - info["transaction_cost"]
            - 2.0 * info["future_volatility"] ** 2
        )
        self.assertAlmostEqual(reward, expected)
        self.assertAlmostEqual(
            info["risk_penalty"], 2.0 * info["future_volatility"] ** 2
        )

    def test_feature_builder_never_receives_future_rows(self):
        calls = []

        def guarded_features(history: pd.DataFrame, as_of_idx: int) -> dict:
            calls.append((len(history), as_of_idx))
            self.assertEqual(len(history), as_of_idx + 1)
            return deterministic_features(history, as_of_idx)

        self.make_environment(feature_builder=guarded_features)

        self.assertEqual(calls, [(3, 2), (5, 4), (7, 6), (9, 8)])

    def test_episode_allows_only_one_step(self):
        env = self.make_environment()
        env.reset(options={"episode_index": 0})
        env.step("hold")

        with self.assertRaises(RuntimeError):
            env.step("hold")

    def test_rejects_invalid_actions_and_episode_indexes(self):
        env = self.make_environment()

        with self.assertRaises(IndexError):
            env.reset(options={"episode_index": len(env)})

        env.reset(options={"episode_index": 0})
        with self.assertRaises(ValueError):
            env.step(len(ACTION_LABELS))


if __name__ == "__main__":
    unittest.main()
