from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class LSTMForecaster(nn.Module):
    hidden_size: int = 128
    num_layers: int = 2
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i in range(self.num_layers):
            x = nn.RNN(nn.OptimizedLSTMCell(features=self.hidden_size))(x)
            if i < self.num_layers - 1 and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

        # take last timestep
        x = x[:, -1, :]
        x = nn.Dense(self.hidden_size // 2)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)
