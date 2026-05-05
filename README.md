# predict

LSTM for forecasting S&P 500 closing prices, with both PyTorch and JAX backends.

Takes 60 days of OHLCV data as input, predicts the next day's close. Features get min-max scaled, fed through a 2-layer LSTM with dropout, then a small feedforward head outputs the prediction. Train/val/test splits are chronological to avoid data leakage.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# pytorch (default)
python main.py

# jax backend
python main.py --backend jax

# tweak hyperparams
python main.py --backend jax --epochs 50 --lr 0.0005 --seq-len 90 --hidden-size 256
```

Saves model checkpoint and prediction plot to `outputs/`.

## Project layout

```
predict/
  config.py        hyperparameters
  data.py          fetch from yahoo finance, scale, build sequences
  model.py         PyTorch LSTM
  model_jax.py     Flax/JAX LSTM (same architecture)
  train.py         PyTorch training loop w/ early stopping
  train_jax.py     JAX/Optax training loop w/ early stopping
  evaluate.py      metrics + plotting (shared across backends)
main.py            CLI with --backend flag
tests/             pytest
```

Both backends implement the same architecture and training procedure (early stopping, gradient clipping, chronological splits). The data pipeline is shared.

## Config

Defaults in `predict/config.py`:

| Param | Default | |
|---|---|---|
| `sequence_length` | 60 | trading days per window |
| `hidden_size` | 128 | LSTM hidden dim |
| `num_layers` | 2 | stacked LSTMs |
| `dropout` | 0.2 | |
| `learning_rate` | 1e-3 | Adam |
| `patience` | 10 | early stopping epochs |
| `train_ratio` | 0.7 | chronological split |
| `val_ratio` | 0.15 | |

## Notes

Splits are strictly chronological (no shuffling across time). Gradient norms are clipped at 1.0. Best checkpoint by validation loss gets restored before test eval.
