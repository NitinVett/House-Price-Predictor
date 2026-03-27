from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def summarize_metrics(portfolio_values: list[float], trade_count: int) -> dict[str, float]:
    values = np.asarray(portfolio_values, dtype=float)
    returns = np.diff(np.log(np.maximum(values, 1e-8)))
    total_return = float(values[-1] / values[0] - 1.0)
    if returns.size > 1 and np.std(returns) > 0:
        sharpe = float((returns.mean() / returns.std()) * np.sqrt(12))
    else:
        sharpe = 0.0
    running_max = np.maximum.accumulate(values)
    max_drawdown = float(np.min(values / running_max - 1.0))
    return {
        "total_return": total_return,
        "annualized_sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "trade_count": int(trade_count),
        "final_portfolio_value": float(values[-1]),
    }


def evaluate_model(
    model: Any,
    data: pd.DataFrame,
    *,
    initial_cash: float,
    transaction_cost: float,
) -> dict[str, Any]:
    from .environment import EnvironmentConfig, TorontoTradingEnv

    env = TorontoTradingEnv(
        data,
        config=EnvironmentConfig(
            initial_cash=initial_cash,
            transaction_cost=transaction_cost,
            random_start=False,
        ),
    )
    observation, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

    portfolio_values = [entry["portfolio_value"] for entry in env.history]
    metrics = summarize_metrics(portfolio_values, env.trade_count)
    metrics["equity_curve"] = [
        {
            "date": pd.Timestamp(entry["date"]).strftime("%Y-%m-%d"),
            "portfolio_value": float(entry["portfolio_value"]),
            "position": int(entry["position"]),
            "action": int(entry["action"]),
        }
        for entry in env.history
    ]
    return metrics


def buy_and_hold_metrics(
    data: pd.DataFrame,
    *,
    initial_cash: float,
    transaction_cost: float,
) -> dict[str, float]:
    first_price = float(data.iloc[0]["target_price"])
    units = (initial_cash * (1.0 - transaction_cost)) / first_price
    values = (data["target_price"].astype(float) * units).tolist()
    values.insert(0, initial_cash)
    return summarize_metrics(values, trade_count=1)


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def maybe_save_equity_curve_plot(metrics: dict[str, Any], path: str | Path) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    curve = metrics.get("equity_curve", [])
    if not curve:
        return None

    dates = [entry["date"] for entry in curve]
    values = [entry["portfolio_value"] for entry in curve]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, values, linewidth=2)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return str(output_path)
