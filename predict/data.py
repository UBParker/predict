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
    return df[list(cfg.raw_columns)].dropna()


def engineer_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Derive non-redundant features from raw OHLCV data.

    Raw OHLC prices are ~1.0 correlated with each other. Instead we keep
    Close and Volume as-is and replace High/Low/Open with:
      - Range: High - Low (intraday volatility)
      - Intraday_Return: (Close - Open) / Open (intraday direction)
    """
    out = pd.DataFrame(index=df.index)
    out["Close"] = df["Close"]
    out["Volume"] = df["Volume"]
    out["Range"] = df["High"] - df["Low"]
    out["Intraday_Return"] = (df["Close"] - df["Open"]) / df["Open"]
    return out[list(cfg.feature_columns)]


def build_sequences(
    data: np.ndarray, targets: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding window: each sample is seq_len days of features, target is next day's close."""
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i : i + seq_len])
        ys.append(targets[i + seq_len])
    return np.array(xs), np.array(ys)


def prepare_arrays(cfg: Config) -> dict:
    """Fetch, scale, split into numpy arrays. Backend-agnostic."""
    raw = fetch_market_data(cfg)
    df = engineer_features(raw, cfg)

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    features_scaled = feature_scaler.fit_transform(df.values)
    # separate scaler for target so we can invert predictions back to USD
    targets_scaled = target_scaler.fit_transform(df[[cfg.target_column]].values).ravel()

    X, y = build_sequences(features_scaled, targets_scaled, cfg.sequence_length)

    n = len(X)
    train_end = int(n * cfg.train_ratio)
    val_end = int(n * (cfg.train_ratio + cfg.val_ratio))

    return {
        "train": (X[:train_end].astype(np.float32), y[:train_end].astype(np.float32)),
        "val": (X[train_end:val_end].astype(np.float32), y[train_end:val_end].astype(np.float32)),
        "test": (X[val_end:].astype(np.float32), y[val_end:].astype(np.float32)),
        "target_scaler": target_scaler,
        "test_dates": df.index[val_end + cfg.sequence_length :],
    }


def prepare_data(
    cfg: Config,
) -> tuple[DataLoader, DataLoader, DataLoader, MinMaxScaler, MinMaxScaler, pd.DatetimeIndex]:
    """Fetch data, scale, split chronologically, return DataLoaders + scalers."""
    arrays = prepare_arrays(cfg)

    loaders = {}
    for name in ("train", "val", "test"):
        X_split, y_split = arrays[name]
        tensors = TensorDataset(
            torch.tensor(X_split, dtype=torch.float32),
            torch.tensor(y_split, dtype=torch.float32),
        )
        loaders[name] = DataLoader(
            tensors, batch_size=cfg.batch_size, shuffle=(name == "train")
        )

    return loaders["train"], loaders["val"], loaders["test"], None, arrays["target_scaler"], arrays["test_dates"]
