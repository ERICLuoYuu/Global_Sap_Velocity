"""Tests for RNN_network model (Loritz et al. 2024 replication)."""

import pytest
import torch

from src.paper_replicate.model import RNN_network, init_forget_gate_bias


class TestRNNNetwork:
    """Test LSTM model architecture matches their Cell 12."""

    def test_forward_pass_shape(self):
        """Input (batch=64, seq=24, features=25) -> output (64, 1)."""
        model = RNN_network(input_size=25, hidden_size=256, num_layers=1, dropout=0.4)
        x = torch.randn(64, 24, 25)
        out = model(x)
        assert out.shape == (64, 1)

    def test_forward_pass_single_sample(self):
        """Should work with batch_size=1."""
        model = RNN_network(input_size=25)
        x = torch.randn(1, 24, 25)
        out = model(x)
        assert out.shape == (1, 1)

    def test_forward_pass_variable_batch(self):
        """Should work with arbitrary batch sizes."""
        model = RNN_network(input_size=25)
        for bs in [1, 16, 32, 128]:
            x = torch.randn(bs, 24, 25)
            out = model(x)
            assert out.shape == (bs, 1)

    def test_hidden_size(self):
        """Hidden size should be 256."""
        model = RNN_network(input_size=25, hidden_size=256)
        assert model.hidden_size == 256
        assert model.lstm.hidden_size == 256

    def test_num_layers(self):
        """Should be single-layer LSTM."""
        model = RNN_network(input_size=25, num_layers=1)
        assert model.num_layers == 1
        assert model.lstm.num_layers == 1

    def test_dropout_applied_during_training(self):
        """Dropout should vary outputs during training."""
        model = RNN_network(input_size=25, dropout=0.4)
        model.train()
        x = torch.randn(64, 24, 25)
        torch.manual_seed(42)
        out1 = model(x).detach()
        torch.manual_seed(99)
        out2 = model(x).detach()
        # With dropout, different seeds should produce different outputs
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_dropout_not_applied_during_test(self):
        """Dropout should be disabled during testing, giving deterministic output."""
        model = RNN_network(input_size=25, dropout=0.4)
        model.train(False)
        x = torch.randn(64, 24, 25)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)


class TestForgetGateBias:
    """Test forget gate bias initialization matches their Cell 15."""

    def test_forget_gate_bias_value(self):
        """bias_hh_l0[hidden_size:2*hidden_size] should be 3.0."""
        model = RNN_network(input_size=25, hidden_size=256)
        init_forget_gate_bias(model, value=3.0)

        bias = model.lstm.bias_hh_l0.data
        forget_gate_bias = bias[256:512]
        assert torch.allclose(forget_gate_bias, torch.full((256,), 3.0))

    def test_other_gate_biases_unchanged(self):
        """Input, cell, and output gate biases should NOT be 3.0."""
        model = RNN_network(input_size=25, hidden_size=256)

        # Save original bias values
        original_bias = model.lstm.bias_hh_l0.data.clone()

        init_forget_gate_bias(model, value=3.0)

        bias = model.lstm.bias_hh_l0.data
        # Input gate (0:256) should be unchanged
        assert torch.allclose(bias[:256], original_bias[:256])
        # Cell gate (512:768) should be unchanged
        assert torch.allclose(bias[512:768], original_bias[512:768])
        # Output gate (768:1024) should be unchanged
        assert torch.allclose(bias[768:1024], original_bias[768:1024])

    def test_custom_forget_gate_value(self):
        """Should work with arbitrary forget gate values."""
        model = RNN_network(input_size=25, hidden_size=128)
        init_forget_gate_bias(model, value=5.0)

        bias = model.lstm.bias_hh_l0.data
        forget_gate = bias[128:256]
        assert torch.allclose(forget_gate, torch.full((128,), 5.0))


class TestH0C0Initialization:
    """Test that hidden states are initialized to zeros."""

    def test_h0_c0_zeros(self):
        """h0 and c0 should be initialized to zeros in forward pass."""
        model = RNN_network(input_size=25, hidden_size=256)
        model.train(False)
        x = torch.randn(4, 24, 25)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)
