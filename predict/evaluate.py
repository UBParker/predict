from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from predict.model import LSTMForecaster

logger = logging.getLogger(__name__)


def evaluate_model(
    model: LSTMForecaster,
    test_loader: DataLoader,
    target_scaler: MinMaxScaler,
    test_dates: pd.DatetimeIndex,
    device: torch.device,
    output_dir: Path | None = None,
) -> dict[str, float]:
    """Run on test set, print metrics, optionally save a plot."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch.to(device)).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())

    pred_scaled = np.concatenate(all_preds).reshape(-1, 1)
    true_scaled = np.concatenate(all_targets).reshape(-1, 1)

    pred_prices = target_scaler.inverse_transform(pred_scaled).ravel()
    true_prices = target_scaler.inverse_transform(true_scaled).ravel()

    metrics = {
        "mse": mean_squared_error(true_prices, pred_prices),
        "rmse": np.sqrt(mean_squared_error(true_prices, pred_prices)),
        "mae": mean_absolute_error(true_prices, pred_prices),
        "r2": r2_score(true_prices, pred_prices),
    }

    logger.info(
        f"Test metrics | RMSE: ${metrics['rmse']:.2f} | MAE: ${metrics['mae']:.2f} | R²: {metrics['r2']:.4f}"
    )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        _plot_predictions(true_prices, pred_prices, test_dates, metrics, output_dir)

    return metrics


def _plot_predictions(
    true_prices: np.ndarray,
    pred_prices: np.ndarray,
    dates: pd.DatetimeIndex,
    metrics: dict[str, float],
    output_dir: Path,
) -> None:
    n = min(len(dates), len(true_prices))
    dates = dates[:n]
    true_prices = true_prices[:n]
    pred_prices = pred_prices[:n]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(dates, true_prices, label="Actual", linewidth=1.2)
    ax1.plot(dates, pred_prices, label="Predicted", linewidth=1.2, alpha=0.85)
    ax1.set_title(f"S&P 500 Close Price Prediction (RMSE: ${metrics['rmse']:.2f}, R²: {metrics['r2']:.4f})")
    ax1.set_ylabel("Price (USD)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    residuals = true_prices - pred_prices
    ax2.bar(dates, residuals, width=2, alpha=0.6, color="steelblue")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Residual (USD)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "predictions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved prediction plot to {path}")
