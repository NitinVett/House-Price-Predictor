from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .environment import EnvironmentConfig, TorontoTradingEnv
from .evaluation import evaluate_model, save_metrics

try:
    from stable_baselines3 import A2C, DQN, PPO
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise ModuleNotFoundError(
        "stable-baselines3 is required for toronto_rl.training. Install project dependencies first."
    ) from exc


MODEL_BUILDERS = {
    "dqn": DQN,
    "ppo": PPO,
    "a2c": A2C,
}


@dataclass(frozen=True)
class TrainConfig:
    timesteps: int = 10_000
    initial_cash: float = 100_000.0
    transaction_cost: float = 0.001
    seed: int = 7


def build_model(model_name: str, env: TorontoTradingEnv, seed: int) -> Any:
    model_name = model_name.lower()
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported model {model_name!r}. Choose from {sorted(MODEL_BUILDERS)}.")

    common_kwargs = {"policy": "MlpPolicy", "env": env, "verbose": 0, "seed": seed}
    if model_name == "dqn":
        return DQN(buffer_size=5_000, learning_starts=100, **common_kwargs)
    if model_name == "ppo":
        return PPO(n_steps=64, batch_size=64, **common_kwargs)
    return A2C(n_steps=16, **common_kwargs)


def train_model(
    model_name: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    *,
    output_dir: str | Path,
    config: TrainConfig | None = None,
) -> tuple[str, dict[str, Any]]:
    train_config = config or TrainConfig()
    env = TorontoTradingEnv(
        train_data,
        config=EnvironmentConfig(
            initial_cash=train_config.initial_cash,
            transaction_cost=train_config.transaction_cost,
            random_start=True,
            seed=train_config.seed,
        ),
    )
    model = build_model(model_name, env, train_config.seed)
    model.learn(total_timesteps=train_config.timesteps)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / f"{model_name.lower()}_toronto_hpi"
    model.save(model_path)

    metrics = evaluate_model(
        model,
        test_data,
        initial_cash=train_config.initial_cash,
        transaction_cost=train_config.transaction_cost,
    )
    metrics["model_name"] = model_name.lower()
    metrics["model_path"] = str(model_path) + ".zip"
    save_metrics(metrics, output_path / f"{model_name.lower()}_metrics.json")
    return str(model_path) + ".zip", metrics


def load_model(model_name: str, model_path: str | Path, env: TorontoTradingEnv | None = None) -> Any:
    model_name = model_name.lower()
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported model {model_name!r}. Choose from {sorted(MODEL_BUILDERS)}.")
    return MODEL_BUILDERS[model_name].load(str(model_path), env=env)
