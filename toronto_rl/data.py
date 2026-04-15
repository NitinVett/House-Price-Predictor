from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PRICE_COLUMNS = [
    "CompIndex",
    "SFDetachIndex",
    "SFAttachIndex",
    "THouseIndex",
    "ApartIndex",
]


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    test: pd.DataFrame


def build_feature_columns(window_size: int) -> list[str]:
    feature_columns: list[str] = []
    for column in PRICE_COLUMNS:
        for lag in range(window_size):
            feature_columns.append(f"{column}_lag_{lag}_norm")
        feature_columns.extend(
            [
                f"{column}_ma_gap_3",
                f"{column}_ma_gap_6",
                f"{column}_momentum_1",
                f"{column}_momentum_3",
                f"{column}_volatility_3",
            ]
        )
    return feature_columns


def load_toronto_data(
    csv_path: str | Path,
    *,
    location: str = "City of Toronto",
    window_size: int = 6,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.loc[df["Location"] == location].copy()
    if df.empty:
        raise ValueError(f"No rows found for location {location!r} in {csv_path}.")

    df["Date"] = pd.to_datetime(df["Date"])
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    grouped = (
        df.groupby("Date", as_index=False)[numeric_columns]
        .mean()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    feature_columns = build_feature_columns(window_size)
    returns = grouped[PRICE_COLUMNS].pct_change()
    for column in PRICE_COLUMNS:
        for lag in range(window_size):
            base = grouped[column].shift(window_size - 1)
            grouped[f"{column}_lag_{lag}_norm"] = grouped[column].shift(lag) / base - 1.0

        ma_3 = grouped[column].rolling(3).mean()
        ma_6 = grouped[column].rolling(6).mean()
        grouped[f"{column}_ma_gap_3"] = grouped[column] / ma_3 - 1.0
        grouped[f"{column}_ma_gap_6"] = grouped[column] / ma_6 - 1.0
        grouped[f"{column}_momentum_1"] = grouped[column].pct_change(1)
        grouped[f"{column}_momentum_3"] = grouped[column].pct_change(3)
        grouped[f"{column}_volatility_3"] = returns[column].rolling(3).std().fillna(0.0)

    warmup = max(window_size - 1, 5)
    prepared = grouped.iloc[warmup:].reset_index(drop=True)
    prepared = prepared.dropna(subset=feature_columns).reset_index(drop=True)
    prepared["target_price"] = prepared["CompIndex"]
    keep_columns = ["Date", *PRICE_COLUMNS, *feature_columns, "target_price"]
    prepared = prepared.loc[:, keep_columns].copy()

    prepared.attrs["feature_columns"] = feature_columns
    prepared.attrs["window_size"] = window_size
    return prepared


def split_dataset(df: pd.DataFrame, *, test_ratio: float = 0.2) -> DatasetSplit:
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1.")
    split_index = max(1, int(len(df) * (1 - test_ratio)))
    split_index = min(split_index, len(df) - 1)
    train = df.iloc[:split_index].reset_index(drop=True)
    test = df.iloc[split_index:].reset_index(drop=True)
    train.attrs = dict(df.attrs)
    test.attrs = dict(df.attrs)
    return DatasetSplit(train=train, test=test)
