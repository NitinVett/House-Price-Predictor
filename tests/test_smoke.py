from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from toronto_rl.data import load_toronto_data, split_dataset
from toronto_rl.evaluation import buy_and_hold_metrics


HAS_GYM = importlib.util.find_spec("gymnasium") is not None
HAS_SB3 = importlib.util.find_spec("stable_baselines3") is not None


@unittest.skipUnless(HAS_GYM and HAS_SB3, "gymnasium and stable-baselines3 are not installed")
class SmokeTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        data = load_toronto_data("MLS.csv", window_size=6)
        cls.split = split_dataset(data, test_ratio=0.2)

    def test_buy_and_hold_baseline_returns_metrics(self) -> None:
        metrics = buy_and_hold_metrics(
            self.split.test,
            initial_cash=100_000.0,
            transaction_cost=0.001,
        )
        self.assertIn("annualized_sharpe", metrics)
        self.assertIn("max_drawdown", metrics)

    def test_short_training_run_for_each_model(self) -> None:
        from toronto_rl.training import TrainConfig, load_model, train_model

        with tempfile.TemporaryDirectory() as tmpdir:
            for model_name in ("dqn", "ppo", "a2c"):
                model_path, metrics = train_model(
                    model_name,
                    self.split.train,
                    self.split.test,
                    output_dir=Path(tmpdir),
                    config=TrainConfig(timesteps=64),
                )
                self.assertTrue(Path(model_path).exists())
                self.assertIn("final_portfolio_value", metrics)
                reloaded = load_model(model_name, model_path)
                self.assertIsNotNone(reloaded)


if __name__ == "__main__":
    unittest.main()
