"""LSTM model for Loritz et al. (2024) replication.

Faithful port of their Cell 12 (RNN_network) and Cell 15 (forget gate init).
"""

import torch
import torch.nn as nn


class RNN_network(nn.Module):
    """LSTM model matching their published architecture exactly.

    Architecture: LSTM(input, 256, 1 layer) -> last timestep -> dropout -> linear(1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
        )
        self.linear_mean = nn.Linear(
            in_features=hidden_size,
            out_features=1,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        device = x.device

        # Initialize hidden states to zeros (their Cell 12)
        h0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device
        ).requires_grad_()
        c0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device
        ).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Last timestep (many-to-one)
        out = self.dropout(out)
        out = self.linear_mean(out)
        return out


def init_forget_gate_bias(network: RNN_network, value: float = 3.0) -> None:
    """Initialize LSTM forget gate bias to given value (their Cell 15).

    The forget gate bias is the second quarter of bias_hh_l0:
    bias_hh_l0[hidden_size : 2*hidden_size] = value
    """
    hidden_size = network.hidden_size
    network.lstm.bias_hh_l0.data[hidden_size : 2 * hidden_size] = value
