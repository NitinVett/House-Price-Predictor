from __future__ import annotations

import argparse
import json
from pathlib import Path

from toronto_rl.data import load_toronto_data, split_dataset
from toronto_rl.evaluation import buy_and_hold_metrics, evaluate_model, maybe_save_equity_curve_plot


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-path", default="MLS.csv", help="Path to the MLS.csv dataset.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for models and metrics.")
    parser.add_argument("--window-size", type=int, default=6, help="Feature window size.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Holdout ratio for evaluation.")
    parser.add_argument("--timesteps", type=int, default=10_000, help="Training timesteps per model.")
    parser.add_argument("--transaction-cost", type=float, default=0.001, help="Proportional transaction fee.")
    parser.add_argument("--initial-cash", type=float, default=100_000.0, help="Starting portfolio cash.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate RL agents on Toronto HPI data.")
    common = argparse.ArgumentParser(add_help=False)
    add_common_arguments(common)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", parents=[common], help="Train one or more RL models.")
    train_parser.add_argument(
        "--models",
        nargs="+",
        default=["dqn", "ppo", "a2c"],
        choices=["dqn", "ppo", "a2c"],
        help="Models to train.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate", parents=[common], help="Evaluate a saved RL model."
    )
    evaluate_parser.add_argument("--model", required=True, choices=["dqn", "ppo", "a2c"])
    evaluate_parser.add_argument("--model-path", required=True, help="Path to the saved model zip file.")

    all_parser = subparsers.add_parser(
        "all", parents=[common], help="Train all baseline models and evaluate them."
    )
    all_parser.add_argument(
        "--models",
        nargs="+",
        default=["dqn", "ppo", "a2c"],
        choices=["dqn", "ppo", "a2c"],
        help="Models to train.",
    )
    return parser


def print_metrics(label: str, metrics: dict[str, float]) -> None:
    summary = {
        "label": label,
        "total_return": round(metrics["total_return"], 4),
        "annualized_sharpe": round(metrics["annualized_sharpe"], 4),
        "max_drawdown": round(metrics["max_drawdown"], 4),
        "trade_count": metrics["trade_count"],
        "final_portfolio_value": round(metrics["final_portfolio_value"], 2),
    }
    print(json.dumps(summary, indent=2))


def run() -> None:
    args = build_parser().parse_args()
    data = load_toronto_data(args.data_path, window_size=args.window_size)
    split = split_dataset(data, test_ratio=args.test_ratio)

    if args.command in {"train", "all"}:
        from toronto_rl.training import TrainConfig, train_model

        for model_name in args.models:
            model_path, metrics = train_model(
                model_name,
                split.train,
                split.test,
                output_dir=Path(args.output_dir),
                config=TrainConfig(
                    timesteps=args.timesteps,
                    initial_cash=args.initial_cash,
                    transaction_cost=args.transaction_cost,
                ),
            )
            print(f"saved_model={model_path}")
            print_metrics(model_name, metrics)
            plot_path = maybe_save_equity_curve_plot(
                metrics,
                Path(args.output_dir) / f"{model_name}_equity_curve.png",
            )
            if plot_path:
                print(f"saved_plot={plot_path}")

        baseline = buy_and_hold_metrics(
            split.test,
            initial_cash=args.initial_cash,
            transaction_cost=args.transaction_cost,
        )
        print_metrics("buy_and_hold", baseline)
        return

    from toronto_rl.environment import EnvironmentConfig, TorontoTradingEnv
    from toronto_rl.training import load_model

    env = TorontoTradingEnv(
        split.test,
        config=EnvironmentConfig(
            initial_cash=args.initial_cash,
            transaction_cost=args.transaction_cost,
            random_start=False,
        ),
    )
    model = load_model(args.model, args.model_path, env=env)
    metrics = evaluate_model(
        model,
        split.test,
        initial_cash=args.initial_cash,
        transaction_cost=args.transaction_cost,
    )
    print_metrics(args.model, metrics)
    plot_path = maybe_save_equity_curve_plot(
        metrics,
        Path(args.output_dir) / f"{args.model}_equity_curve.png",
    )
    if plot_path:
        print(f"saved_plot={plot_path}")


if __name__ == "__main__":
    run()
