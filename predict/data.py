from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from predict.config import Config


def fetch_market_data(cfg: Config) -> pd.DataFrame:
    df = yf.download(cfg.ticker, start=cfg.start_date, end=cfg.end_date, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {cfg.ticker}")
    # yfinance sometimes returns MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df[list(cfg.feature_columns)].dropna()


def build_sequences(
    data: np.ndarray, targets: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding window: each sample is seq_len days of features, target is next day's close."""
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i : i + seq_len])
        ys.append(targets[i + seq_len])
    return np.array(xs), np.array(ys)


def prepare_data(
    cfg: Config,
) -> tuple[DataLoader, DataLoader, DataLoader, MinMaxScaler, MinMaxScaler, pd.DatetimeIndex]:
    """Fetch data, scale, split chronologically, return DataLoaders + scalers."""
    df = fetch_market_data(cfg)

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    features_scaled = feature_scaler.fit_transform(df.values)
    # separate scaler for target so we can invert predictions back to USD
    targets_scaled = target_scaler.fit_transform(df[[cfg.target_column]].values).ravel()

    X, y = build_sequences(features_scaled, targets_scaled, cfg.sequence_length)

    n = len(X)
    train_end = int(n * cfg.train_ratio)
    val_end = int(n * (cfg.train_ratio + cfg.val_ratio))

    splits = {
        "train": (X[:train_end], y[:train_end]),
        "val": (X[train_end:val_end], y[train_end:val_end]),
        "test": (X[val_end:], y[val_end:]),
    }

    loaders = {}
    for name, (X_split, y_split) in splits.items():
        tensors = TensorDataset(
            torch.tensor(X_split, dtype=torch.float32),
            torch.tensor(y_split, dtype=torch.float32),
        )
        loaders[name] = DataLoader(
            tensors, batch_size=cfg.batch_size, shuffle=(name == "train")
        )

    test_dates = df.index[val_end + cfg.sequence_length :]

    return loaders["train"], loaders["val"], loaders["test"], feature_scaler, target_scaler, test_dates
