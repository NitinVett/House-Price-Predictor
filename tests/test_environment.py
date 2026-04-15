from __future__ import annotations

import importlib.util
import unittest

from toronto_rl.data import load_toronto_data


@unittest.skipUnless(importlib.util.find_spec("gymnasium"), "gymnasium is not installed")
class EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_toronto_data("MLS.csv", window_size=6).head(12).reset_index(drop=True)

    def test_invalid_sell_gets_penalized(self) -> None:
        from toronto_rl.environment import EnvironmentConfig, TorontoTradingEnv

        env = TorontoTradingEnv(
            self.data,
            config=EnvironmentConfig(initial_cash=100_000.0, transaction_cost=0.001),
        )
        env.reset(seed=1)
        _, reward, _, _, info = env.step(2)
        self.assertLess(reward, 0.0)
        self.assertEqual(info["trade_count"], 0)
        self.assertEqual(info["position"], 0)

    def test_buy_then_sell_updates_portfolio_state(self) -> None:
        from toronto_rl.environment import EnvironmentConfig, TorontoTradingEnv

        env = TorontoTradingEnv(
            self.data,
            config=EnvironmentConfig(initial_cash=100_000.0, transaction_cost=0.001),
        )
        env.reset(seed=1)
        _, _, _, _, buy_info = env.step(1)
        self.assertEqual(buy_info["position"], 1)
        self.assertEqual(buy_info["trade_count"], 1)

        _, _, _, _, sell_info = env.step(2)
        self.assertEqual(sell_info["position"], 0)
        self.assertEqual(sell_info["trade_count"], 2)


if __name__ == "__main__":
    unittest.main()
