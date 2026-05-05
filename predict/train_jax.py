from __future__ import annotations

import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from predict.config import Config
from predict.model_jax import LSTMForecaster

logger = logging.getLogger(__name__)


def create_train_state(cfg: Config, rng: jax.Array, input_shape: tuple) -> train_state.TrainState:
    model = LSTMForecaster(
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout_rate=cfg.dropout,
    )
    variables = model.init(rng, jnp.ones(input_shape), training=False)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(cfg.learning_rate, weight_decay=cfg.weight_decay),
    )
    return train_state.TrainState.create(
        apply_fn=model.apply, params=variables["params"], tx=tx
    )


@jax.jit
def _train_step(state, x, y, dropout_rng):
    def loss_fn(params):
        preds = state.apply_fn(
            {"params": params}, x, training=True, rngs={"dropout": dropout_rng}
        )
        return jnp.mean((preds - y) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def _eval_step(state, x, y):
    preds = state.apply_fn({"params": state.params}, x, training=False)
    return jnp.mean((preds - y) ** 2)


def _batches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = False):
    n = len(X)
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        batch_idx = idx[start : start + batch_size]
        yield jnp.array(X[batch_idx]), jnp.array(y[batch_idx])


def train_model_jax(
    cfg: Config,
    train_data: tuple[np.ndarray, np.ndarray],
    val_data: tuple[np.ndarray, np.ndarray],
    checkpoint_dir: Path | None = None,
) -> train_state.TrainState:
    """Train with early stopping on val loss. Returns best state."""
    rng = jax.random.PRNGKey(cfg.seed)
    rng, init_rng = jax.random.split(rng)

    X_train, y_train = train_data
    X_val, y_val = val_data
    input_shape = (1, cfg.sequence_length, len(cfg.feature_columns))

    state = create_train_state(cfg, init_rng, input_shape)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_params = None

    for epoch in range(1, cfg.max_epochs + 1):
        rng, epoch_rng = jax.random.split(rng)
        train_loss = 0.0
        n_samples = 0

        for X_batch, y_batch in _batches(X_train, y_train, cfg.batch_size, shuffle=True):
            rng, dropout_rng = jax.random.split(rng)
            state, loss = _train_step(state, X_batch, y_batch, dropout_rng)
            train_loss += float(loss) * len(X_batch)
            n_samples += len(X_batch)
        train_loss /= n_samples

        # val loss over all batches
        val_loss = 0.0
        val_n = 0
        for X_batch, y_batch in _batches(X_val, y_val, cfg.batch_size):
            loss = _eval_step(state, X_batch, y_batch)
            val_loss += float(loss) * len(X_batch)
            val_n += len(X_batch)
        val_loss /= val_n

        logger.info(f"Epoch {epoch:3d} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_params = jax.tree.map(lambda x: x.copy(), state.params)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={cfg.patience})")
                break

    state = state.replace(params=best_params)

    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        flat, _ = jax.tree_util.tree_flatten(best_params)
        np.savez(checkpoint_dir / "best_model_jax.npz", *[np.array(x) for x in flat])
        logger.info(f"Saved JAX checkpoint to {checkpoint_dir}")

    return state


def predict_jax(
    state: train_state.TrainState,
    X: np.ndarray,
    batch_size: int = 32,
) -> np.ndarray:
    all_preds = []
    for start in range(0, len(X), batch_size):
        x_batch = jnp.array(X[start : start + batch_size])
        preds = state.apply_fn({"params": state.params}, x_batch, training=False)
        all_preds.append(np.array(preds))
    return np.concatenate(all_preds)
