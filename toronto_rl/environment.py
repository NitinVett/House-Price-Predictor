from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise ModuleNotFoundError(
        "gymnasium is required for toronto_rl.environment. Install project dependencies first."
    ) from exc


@dataclass(frozen=True)
class EnvironmentConfig:
    initial_cash: float = 100_000.0
    transaction_cost: float = 0.001
    invalid_action_penalty: float = 0.001
    random_start: bool = False
    seed: int | None = None


class TorontoTradingEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        start_index: int = 0,
        end_index: int | None = None,
        config: EnvironmentConfig | None = None,
    ) -> None:
        super().__init__()
        self.data = data.reset_index(drop=True).copy()
        if feature_columns is None:
            feature_columns = list(data.attrs.get("feature_columns", []))
        self.feature_columns = feature_columns
        if not self.feature_columns:
            raise ValueError("feature_columns cannot be empty.")
        self.config = config or EnvironmentConfig()
        self.start_index = start_index
        self.end_index = len(self.data) - 1 if end_index is None else end_index
        if self.end_index <= self.start_index:
            raise ValueError("end_index must be greater than start_index.")

        obs_size = len(self.feature_columns) + 3
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(self.config.seed)
        self._episode_start = self.start_index
        self._index = self.start_index
        self.cash = self.config.initial_cash
        self.units = 0.0
        self.position = 0
        self.last_action = 0
        self.trade_count = 0
        self.history: list[dict[str, Any]] = []

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        max_start = self.end_index - 1
        if self.config.random_start and max_start > self.start_index:
            self._episode_start = int(self._rng.integers(self.start_index, max_start))
        else:
            self._episode_start = self.start_index

        self._index = self._episode_start
        self.cash = self.config.initial_cash
        self.units = 0.0
        self.position = 0
        self.last_action = 0
        self.trade_count = 0
        self.history = [
            {
                "step": 0,
                "date": self._current_row()["Date"],
                "portfolio_value": self._portfolio_value(self._current_price()),
                "position": self.position,
                "action": self.last_action,
            }
        ]
        return self._observation(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._index >= self.end_index:
            raise RuntimeError("Episode is done. Call reset().")

        action = int(action)
        current_price = self._current_price()
        current_value = self._portfolio_value(current_price)
        penalty = 0.0

        if action == 1:
            if self.position == 0:
                self.units = (self.cash * (1.0 - self.config.transaction_cost)) / current_price
                self.cash = 0.0
                self.position = 1
                self.trade_count += 1
            else:
                penalty = self.config.invalid_action_penalty
        elif action == 2:
            if self.position == 1:
                self.cash = self.units * current_price * (1.0 - self.config.transaction_cost)
                self.units = 0.0
                self.position = 0
                self.trade_count += 1
            else:
                penalty = self.config.invalid_action_penalty

        self.last_action = action
        self._index += 1
        next_price = self._current_price()
        next_value = self._portfolio_value(next_price)
        reward = float(np.log(max(next_value, 1e-8) / max(current_value, 1e-8)) - penalty)
        terminated = self._index >= self.end_index

        self.history.append(
            {
                "step": len(self.history),
                "date": self._current_row()["Date"],
                "portfolio_value": next_value,
                "position": self.position,
                "action": action,
                "reward": reward,
            }
        )

        info = {
            "portfolio_value": next_value,
            "cash": self.cash,
            "units": self.units,
            "position": self.position,
            "trade_count": self.trade_count,
            "date": self._current_row()["Date"],
        }
        return self._observation(), reward, terminated, False, info

    def _current_row(self) -> pd.Series:
        return self.data.iloc[self._index]

    def _current_price(self) -> float:
        return float(self._current_row()["target_price"])

    def _portfolio_value(self, price: float) -> float:
        return float(self.cash + self.units * price)

    def _observation(self) -> np.ndarray:
        row = self._current_row()
        features = row[self.feature_columns].to_numpy(dtype=np.float32)
        portfolio_value = max(self._portfolio_value(self._current_price()), 1e-8)
        portfolio_state = np.array(
            [
                self.cash / portfolio_value,
                float(self.position),
                float(self.last_action),
            ],
            dtype=np.float32,
        )
        return np.concatenate([features, portfolio_state]).astype(np.float32)
