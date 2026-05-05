import jax
import jax.numpy as jnp
import numpy as np

from predict.config import Config
from predict.model_jax import LSTMForecaster


def test_jax_model_forward_shape():
    cfg = Config(hidden_size=32, num_layers=1, dropout=0.0)
    model = LSTMForecaster(hidden_size=32, num_layers=1, dropout_rate=0.0)
    rng = jax.random.PRNGKey(0)
    batch = jnp.ones((4, cfg.sequence_length, len(cfg.feature_columns)))
    variables = model.init(rng, batch, training=False)
    out = model.apply(variables, batch, training=False)
    assert out.shape == (4,)


def test_jax_model_deterministic():
    model = LSTMForecaster(hidden_size=32, num_layers=1, dropout_rate=0.0)
    rng = jax.random.PRNGKey(0)
    batch = jnp.ones((2, 10, 5))
    variables = model.init(rng, batch, training=False)
    out1 = model.apply(variables, batch, training=False)
    out2 = model.apply(variables, batch, training=False)
    np.testing.assert_array_equal(np.array(out1), np.array(out2))
