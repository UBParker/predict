from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    ticker: str = "^GSPC"
    start_date: str = "2010-01-01"
    end_date: str = "2025-12-31"
    feature_columns: tuple[str, ...] = ("Close", "Volume", "High", "Low", "Open")
    target_column: str = "Close"

    sequence_length: int = 60
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 10

    seed: int = 42
