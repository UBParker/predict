import numpy as np
import torch

from predict.config import Config
from predict.data import build_sequences
from predict.model import LSTMForecaster


def test_build_sequences_shapes():
    data = np.random.randn(100, 5)
    targets = np.random.randn(100)
    X, y = build_sequences(data, targets, seq_len=10)
    assert X.shape == (90, 10, 5)
    assert y.shape == (90,)


def test_build_sequences_values():
    data = np.arange(20).reshape(20, 1).astype(float)
    targets = np.arange(20).astype(float)
    X, y = build_sequences(data, targets, seq_len=5)
    np.testing.assert_array_equal(X[0], [[0], [1], [2], [3], [4]])
    assert y[0] == 5.0
    np.testing.assert_array_equal(X[-1], [[14], [15], [16], [17], [18]])
    assert y[-1] == 19.0


def test_model_forward_shape():
    cfg = Config(hidden_size=32, num_layers=1, dropout=0.0)
    model = LSTMForecaster(cfg)
    batch = torch.randn(4, cfg.sequence_length, len(cfg.feature_columns))
    out = model(batch)
    assert out.shape == (4,)


def test_model_gradient_flow():
    cfg = Config(hidden_size=32, num_layers=1, dropout=0.0)
    model = LSTMForecaster(cfg)
    batch = torch.randn(2, cfg.sequence_length, len(cfg.feature_columns))
    target = torch.randn(2)
    loss = torch.nn.MSELoss()(model(batch), target)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
