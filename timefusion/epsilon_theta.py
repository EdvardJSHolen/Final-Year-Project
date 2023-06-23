# Library imports
import torch
import math

# Module imports
from torch import nn, Tensor


class DiffusionEmbedding(nn.Module):
    # This class borrows code from the PyTorch implementation of the Transformer model

    def __init__(self, num_sines: int, out_dim: int, device: torch.device, diff_steps: int = 100):
        super().__init__()

        # Look up table for sine encodings
        steps = torch.arange(diff_steps, device = device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_sines, 2, device=device) * (-math.log(10000.0) / num_sines))
        pe = torch.zeros(diff_steps, num_sines, device = device)
        pe[:, 0::2] = torch.sin(steps * div_term)
        pe[:, 1::2] = torch.cos(steps * div_term)
        self.register_buffer('pe', pe)

        # Upsampling network
        self.linear1 = nn.Linear(num_sines, out_dim, device = device)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(out_dim, out_dim, device = device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor with shape [batch_size] giving the diffusion step
        Returns:
            Tensor with shape [batch_size, out_dim] giving the diffusion embedding
        """
        x = self.pe[x]
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x


class ScaleLayer(nn.Module):

    def __init__(self, dim: int, device: torch.device):
        super().__init__()
        self.scales = torch.nn.Parameter(torch.empty(dim,device=device))
        self.bias = torch.nn.Parameter(torch.empty(dim,device=device))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scales + self.bias
    

class ResidualBlock(nn.Module):

    def __init__(self, residual_size: int, hidden_size: int, device: torch.device):
        super().__init__()

        self.linear1 = nn.Linear(residual_size, hidden_size, device = device)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, residual_size, device=device)
        self.tanh = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        
        #x = self.tanh(x)
        x_ = self.linear1(x)
        x_ = self.relu(x_)
        x_ = self.linear2(x_)
        x = self.tanh(x + x_)
        #x = x + x_
        return x


class EpsilonTheta(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        rnn_layers: int = 2,
        rnn_hidden: int = 40,
        residual_layers: int = 2,
        residual_hidden: int = 100,
        dropout: float = 0.0,
        diff_steps: int = 100,
        device: torch.device = torch.device("cpu"),
        **kwargs
    ):
        """
        Args:
            input_size: Number of input features, i.e. time-series dimension + convariates dimension
            output_size: Number of output features, i.e. time-series dimension
            rnn_layers: Number of RNN layers
            rnn_hidden: Size of RNN hidden state
            residual_layers: Number of residual layers
            residual_size: Size of input and output of residual layers
            residual_hidden: Size of hidden layer in residual layers
            diff_steps: Number of diffusion steps
            device: Device to use for computation
        """

        # Init base class
        super().__init__()

        # Embedding for diffusion steps
        self.embedding = DiffusionEmbedding(
            num_sines = 32,
            out_dim = rnn_hidden,
            device = device,
            diff_steps = diff_steps
        )

        # Instantiate rnn network
        self.rnn = nn.LSTM(
            input_size = input_size,
            hidden_size = rnn_hidden,
            num_layers = rnn_layers,
            dropout = dropout,
            batch_first = True,
            device = device
        )

        # Add residual layers
        layers = []

        residual_size = rnn_hidden + output_size
        if kwargs.get("residual_scaler", False):
            layers.append(ScaleLayer(residual_size, device = device))
        else:
            layers.append(nn.Linear(residual_size, residual_size, device = device))
        #layers.pop()

        for _ in range(residual_layers):
            layers.append(ResidualBlock(residual_size, residual_hidden, device))

        layers.append(nn.Linear(residual_size, output_size, device = device))

        self.residuals = nn.Sequential(*layers)

        
    def forward(self, x: Tensor, n: Tensor, context: Tensor = None, h: Tensor = None) -> Tensor:

        assert (not context is None) or (not h is None), "Either context or hidden state must be provided"

        if h is None:
            context = context.permute((0,2,1))
            h, _  = self.rnn(context)

        _n = self.embedding(n - 1)

        x = self.residuals(torch.cat((x, h[:,-1] + _n), dim = 1))

        return x, h
