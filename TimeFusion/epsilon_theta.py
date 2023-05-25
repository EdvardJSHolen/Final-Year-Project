# Library imports
import torch

# Module imports
from torch import nn, Tensor

import math

class DiffusionEmbedding(nn.Module):

    def __init__(self, dim: int, proj_dim: int, device: torch.device, max_steps: int = 500):
        super().__init__()

        # Look up table for sine encodings
        step = torch.arange(max_steps, device = device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_steps, dim, device = device)
        pe[:, 0::2] = torch.sin(step * div_term)
        pe[:, 1::2] = torch.cos(step * div_term)
        self.register_buffer('pe', pe)

        # FC network
        self.projection1 = nn.Linear(dim, proj_dim,device=device)
        self.tanh1 = nn.Tanh()
        self.projection2 = nn.Linear(proj_dim, proj_dim,device=device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor with shape [batch_size] giving the diffusion step
        """
        x = self.pe[x]
        x = self.projection1(x)
        x = self.tanh1(x)
        x = self.projection2(x)
        return x
    
class ResidualBlock(nn.Module):

    def __init__(self, input_size: int, output_size: int, hidden_size: int, device: torch.device):
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size, device = device)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size, device=device)
        self.tanh = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        
        x_ = self.linear1(x)
        x_ = self.relu(x_)
        x_ = self.linear2(x_)
        x = self.tanh(x + x_)
        return x

class EpsilonTheta(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        rnn_layers: int = 2,
        rnn_hidden: int = 40,
        autoencoder_layers: int = 1,
        autoencoder_latent: int = 40,
        activation_fn: nn.Module = nn.ReLU(),
        diff_steps: int = 100,
        device: torch.device = torch.device("cpu")
    ):
        """
        Args:
            input_size: Number of input features, i.e. time-series dimension + convariates dimension
            output_size: Number of output features, i.e. time-series dimension
            rnn_layers: Number of RNN layers
            rnn_hidden: Size of RNN hidden state
            autoencoder_layers: Number of scaling layers in autoencoder, total number of layers is 2*autoencoder_layers
            autoencoder_latent: Size of autoencoder latent space
            activation_fn: Activation function used in autoencoder
            diff_steps: Number of diffusion steps
        """

        # Init base class
        super().__init__()

        # Diffusion embedding
        #self.embedding = nn.Embedding(
        #    num_embeddings = diff_steps, 
        #     #embedding_dim = output_size, 
        #    embedding_dim = 32,
        #    device = device
        #)

        # # Set embedding weights to sine waves
        # embedding_weights = torch.zeros((self.embedding.weight.shape), device = device)
        # ## THIS IS WHERE I LEFT OFF
        # print(self.embedding.weight.shape)
        # self.embedding.weight = nn.Parameter(torch.ones_like(self.embedding.weight))
        self.embedding = DiffusionEmbedding(
            dim = 32,
            proj_dim = rnn_hidden,
            device= device
        )


        # Instantiate rnn network
        self.rnn = nn.LSTM(
            input_size = input_size,
            hidden_size = rnn_hidden,
            num_layers = rnn_layers,
            batch_first = True,
            device = device
        )

        ### Create autoencoder ###
        layers = []
        
        layers.append(nn.Linear(rnn_hidden + output_size,rnn_hidden + output_size,device=device))
        layers.append(ResidualBlock(rnn_hidden + output_size,rnn_hidden + output_size,100,device))
        layers.append(ResidualBlock(rnn_hidden + output_size,rnn_hidden + output_size,100,device))
        layers.append(ResidualBlock(rnn_hidden + output_size,rnn_hidden + output_size,100,device))
        layers.append(ResidualBlock(rnn_hidden + output_size,rnn_hidden + output_size,100,device))
        layers.append(nn.Linear(rnn_hidden + output_size,output_size,device=device))

        # Downscaling layers
        #factor = autoencoder_latent / (rnn_hidden + output_size)
        #for i in range(autoencoder_layers):
        #    in_features = round((rnn_hidden + output_size)*(factor)**(i/autoencoder_layers))
        #    out_features = round((rnn_hidden + output_size)*(factor)**((i + 1)/autoencoder_layers))
        #    layers.append(nn.Linear(in_features, out_features, device = device))
        #    layers.append(activation_fn)

        # Upscaling layers
        #factor = output_size / autoencoder_latent
        #for i in range(autoencoder_layers):
        #    in_features = round((autoencoder_latent)*(factor)**(i/autoencoder_layers))
        #    out_features = round((autoencoder_latent)*(factor)**((i + 1)/autoencoder_layers))
        #    layers.append(nn.Linear(in_features, out_features, device = device))
        #    layers.append(activation_fn)

        # Remove last activation
        #layers.pop()
        self.autoencoder = nn.Sequential(*layers)
        
    def forward(self, x: Tensor, n: Tensor, context: Tensor = None, h: Tensor = None) -> Tensor:

        assert (not context is None) or (not h is None), "Either context or hidden state must be provided"

        if h is None:
            context = context.permute((0,2,1))
            h, _  = self.rnn(context)

        _n = self.embedding(n - 1)
        #_x = x + _n

        _x = x
        
        _x = torch.cat((_x, h[:,-1] + _n), dim = 1)
        _x = self.autoencoder(_x)

        return _x, h


# TODO:
# 1. Try to initialize the embedding with the sine waves
# 2. Try to initialize the embedding with the sine waves and train the embedding
# 3. Try to initialize the embedding with the sine waves and train the embedding with linear layers
# 4. Try to increase the size of the latent dimension -> Gave better performance
# 5. Try to implement the same network as before but with this new framework and check that I get equivalent results.
# 6. Trying concatenation rather than addition