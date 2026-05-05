#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from predict.config import Config
from predict.data import prepare_data
from predict.evaluate import evaluate_model
from predict.train import train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an LSTM to forecast S&P 500 closing prices.")
    parser.add_argument("--epochs", type=int, default=None, help="Override max training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length")
    parser.add_argument("--hidden-size", type=int, default=None, help="Override LSTM hidden size")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for results")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving the prediction plot")
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.max_epochs = args.epochs
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.seq_len is not None:
        cfg.sequence_length = args.seq_len
    if args.hidden_size is not None:
        cfg.hidden_size = args.hidden_size

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Preparing data...")
    train_loader, val_loader, test_loader, _, target_scaler, test_dates = prepare_data(cfg)
    logger.info(
        f"Data splits | train: {len(train_loader.dataset)} | val: {len(val_loader.dataset)} | test: {len(test_loader.dataset)}"
    )

    logger.info("Training model...")
    model = train_model(cfg, train_loader, val_loader, device, checkpoint_dir=args.output_dir / "checkpoints")

    logger.info("Evaluating on test set...")
    output_dir = None if args.no_plot else args.output_dir
    evaluate_model(model, test_loader, target_scaler, test_dates, device, output_dir=output_dir)


if __name__ == "__main__":
    main()
