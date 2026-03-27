# Toronto HPI RL Baseline

This project trains DQN, PPO, and A2C agents on deduped monthly `City of Toronto` HPI data from `MLS.csv`. The agent trades a single synthetic Toronto market with `HOLD`, `BUY`, and `SELL` actions while using five Toronto price-index series as state inputs.

## Setup

Install the project dependencies:

```bash
pip install -e .
```

## Commands

Train all three baseline agents and evaluate them on the held-out tail of the series:

```bash
python main.py all --data-path MLS.csv --timesteps 10000 --output-dir artifacts
```

Train a subset of models:

```bash
python main.py train --models dqn ppo --data-path MLS.csv --timesteps 5000
```

Evaluate a saved model:

```bash
python main.py evaluate --model ppo --model-path artifacts/ppo_toronto_hpi.zip
```

## What The Pipeline Does

- Filters `MLS.csv` to `Location == "City of Toronto"`.
- Deduplicates monthly rows by averaging numeric columns per date.
- Builds rolling-window features from `CompIndex`, `SFDetachIndex`, `SFAttachIndex`, `THouseIndex`, and `ApartIndex`.
- Splits the sequence chronologically with the last 20% reserved for evaluation.
- Trains DQN, PPO, and A2C baselines with Stable-Baselines3.
- Reports total return, annualized Sharpe ratio, max drawdown, trade count, and final portfolio value.

## Outputs

Artifacts are written under the configured output directory:

- Saved model checkpoints as `.zip`
- Per-model metrics JSON files
- Optional equity-curve plots when `matplotlib` is available

## Tests

Run the test suite with:

```bash
python -m unittest discover -s tests
```
