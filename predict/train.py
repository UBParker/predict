from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from predict.config import Config
from predict.model import LSTMForecaster

logger = logging.getLogger(__name__)


def train_model(
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    checkpoint_dir: Path | None = None,
) -> LSTMForecaster:
    """Train with early stopping on val loss. Returns best model."""
    model = LSTMForecaster(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_loader.dataset)

        val_loss = _evaluate_loss(model, val_loader, criterion, device)

        logger.info(f"Epoch {epoch:3d} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={cfg.patience})")
                break

    model.load_state_dict(best_state)
    model.to(device)

    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint_dir / "best_model.pt")
        logger.info(f"Saved checkpoint to {checkpoint_dir / 'best_model.pt'}")

    return model


def _evaluate_loss(
    model: LSTMForecaster,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            total_loss += criterion(model(X_batch), y_batch).item() * len(X_batch)
    return total_loss / len(loader.dataset)
