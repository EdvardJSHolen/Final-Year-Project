# Library imports
import torch
import math
import calendar

# Module imports
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from pandas import DatetimeIndex
from typing import List, Callable, Optional, Dict, Tuple

# Relative imports
from data import TimeFusionDataset
from diffusion import Diffuser
from utils.modules import MeanScaler, EarlyStopper

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
        self.projection1 = nn.Linear(dim, proj_dim)
        self.tanh1 = nn.Tanh()
        self.projection2 = nn.Linear(proj_dim, proj_dim)

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

    def __init__(self, x_dim: int, hidden_dim: int, noise_dim: int, device: torch.device):
        super().__init__()

        total_dim = x_dim + hidden_dim + noise_dim

        # FC network
        self.projection1 = nn.Linear(total_dim, total_dim)
        self.projection2 = nn.Linear(total_dim, x_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor, h: Tensor, n: Tensor) -> Tensor:

        out = torch.concat([x,h,n],dim = 1)
        out = self.projection1(out)
        out = self.relu(out)
        out = self.projection2(out)
        out = self.relu(out + x)

        return out


# Class accessible to end-user
class TimeFusion(nn.Module):

    def __init__(
            self,
            input_size: int,
            hidden_size: int = 40,
            recurrent_layers: int = 2,
            dropout: float = 0.1,
            residual_layers: int = 2,
            scaling: bool = False,
            diff_steps: int = 100,
            betas: List[float] = None,
            device: torch.device = torch.device("cpu"),
        ) -> None:
        
        # Init nn.Module base class
        super().__init__()

        ### Set instance variables ###
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_layers = recurrent_layers
        self.residual_layers = residual_layers
        self.scaling = scaling
        self.diff_steps = diff_steps
        self.device = device

        if scaling:
            self.scaler = MeanScaler(
                device = device,
                min_scale = 0.01
            )

        self.diff_embedding = DiffusionEmbedding(
            dim = 32,
            proj_dim = 32,
            device= device
        )

        self.diffuser = Diffuser(
            diff_steps = diff_steps,
            betas = betas,
            device = device
        )
        
        self.rnn = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = recurrent_layers,
            dropout = dropout,
            batch_first = True
        )

        self.residual = nn.ModuleList(
            [
                ResidualBlock(
                    x_dim = input_size,
                    hidden_dim = hidden_size,
                    noise_dim = 32,
                    device = device
                )
                for _ in range(residual_layers)
            ]
        )
        
        self.linear = nn.Linear(
            in_features = input_size,
            out_features = input_size
        )
    
    def forward(self, context: Tensor, x: Tensor, n: Tensor):
        """
        Args:
            context: Tensor of shape (batch size, time-series dim, context length)
            x: Tensor of shape (batch size, time-series dim)
            n: Tensor of shape (batch size)
        """
        context = context.permute((0,2,1))
        h, _  = self.rnn(context)
        n = self.diff_embedding(n)

        for layer in self.residual:
            x = layer(x,h[:,-1],n)

        x = self.linear(x)

        return x

    # Function to train TimeFusion network
    def train_network(self,
            train_loader: DataLoader, 
            epochs: int,
            val_loader: Optional[DataLoader] = None,
            val_metrics: Optional[Dict[str,Callable]] = None,
            loss_function: Callable = nn.MSELoss(),
            optimizer: optim.Optimizer = None,
            lr_scheduler: _LRScheduler = None,
            early_stopper: EarlyStopper = None,
        ) -> None:
        """
        Args:
            train_loader: Generator which provides batched training data.
            epochs: The number of epochs to train for.
            train_loader: Generator which provides batched validation data.
            val_metrics: Name of metrics and callable functions to calculate them which measure the performance of the network.
            loss_function: Function to measure how well predictions match targets.
            optimizer: Optimizer used to update weights.
            lr_scheduler: Learning rate scheduler which modifies learning rate after each epoch.
        """

        # Set the network into training mode
        self.train(True)

        # Set default optimizer
        if optimizer is None:
            optimizer = optim.Adam(params = self.parameters(), lr = 1e-4)

        # Set default validation metrics
        val_metrics = val_metrics | {"val_loss": loss_function}
            

        for epoch in range(1, epochs + 1):

            running_loss = 0
            for i, data in enumerate(train_loader, start = 1):
                context, target = data
                #if self.device == torch.device("mps"):
                context = context.to(self.device)
                target = target.to(self.device)

                if self.scaling:
                    context = self.scaler(context)

                target /= self.scaler.scales

                # Diffuse data
                x, target, n = self.diffuser.diffuse(target)

                # Zero gradients
                optimizer.zero_grad()

                # Forward, loss calculation, backward, optimizer step
                predictions = self.forward(context, x, n)
                #predictions = self.forward(context, None, None)
                loss = loss_function(predictions,target)
                loss.backward()
                optimizer.step()

                # Print training statistics
                running_loss += loss.item()
                average_loss = running_loss / i
                stat_string = "|" + "="*(30*i // len(train_loader)) + " "*(30 - (30*i // len(train_loader))) + f"|  Batch: {i} / {len(train_loader)}, Epoch: {epoch} / {epochs}, Average Loss: {average_loss:.4f}"
                print("\u007F"*512,stat_string,end="\r")

            if lr_scheduler:
                lr_scheduler.step()

            # if val_loader is not None:
            #     with torch.no_grad():
            #         running_loss = {key:0 for key in val_metrics.keys()}
            #         for tokens in val_loader:
            #             #if self.device == torch.device("mps"):
            #             tokens = tokens.to(self.device)

            #             if self.scaling:
            #                 tokens = self.scaler(tokens)

            #             # Diffuse data
            #             tokens, targets = self.diffuser.diffuse(tokens)

            #             # Calculate prediction metrics
            #             predictions = self.forward(tokens)
            #             for key, metric_func in val_metrics.items():
            #                 running_loss[key] += metric_func(predictions,targets).item() 
                    
            #         for metric, value in running_loss.items():
            #             stat_string += f", {metric}: {value / len(val_loader):.4f}"

            #         print("\u007F"*512,stat_string)

            #         if not early_stopper is None:
            #             stop, weights = early_stopper.early_stop(running_loss["val_loss"] / len(val_loader),self)
            #             if stop:
            #                 if not weights is None:
            #                     self.load_state_dict(weights)
            #                 break
            # else:
            #     if not early_stopper is None:
            #             stop, weights = early_stopper.early_stop(average_loss,self)
            #             if stop:
            #                 if not weights is None:
            #                     self.load_state_dict(weights)
            #                 break

                # New line for printing statistics
                print()

        # if (not early_stopper is None) and early_stopper.restore_weights:
        #     self.load_state_dict(early_stopper.best_weights)


    # @torch.no_grad()
    # def sample(
    #     self,
    #     data: TimeFusionDataset,
    #     sample_indices: List[DatetimeIndex],
    #     num_samples: int = 1,
    #     batch_size: int = 64,
    #     timestamp_encodings: List[Callable] = [
    #         lambda x: math.sin(2*math.pi*x.hour / 24),
    #         lambda x: math.sin(2*math.pi*x.weekday() / 7),
    #         lambda x: math.sin(2*math.pi*x.day / calendar.monthrange(x.year, x.month)[1]),
    #         lambda x: math.sin(2*math.pi*x.month / 12),
    #         lambda x: math.cos(2*math.pi*x.hour / 24),
    #         lambda x: math.cos(2*math.pi*x.weekday() / 7),
    #         lambda x: math.cos(2*math.pi*x.day / calendar.monthrange(x.year, x.month)[1]),
    #         lambda x: math.cos(2*math.pi*x.month / 12),
    #     ],
    # ):
        
    #     """
    #     Args:
    #         data: Historical data which is fed as context to the network.
    #         timestamps: Indexes at which to predict future values, length should be prediction length.
    #         num_samples: Number of samples to sample from the network.
    #         batch_size: The number of samples to process at the same time.

    #     Note:
    #         1. For optimal performance, num_samples should be divisible by batch size.
    #         2. The timestamps must be drawn from the same distribution as those from the training data for best performance
    #     """

    #     # Set the network into evaluation mode
    #     self.train(False)
        

    #     # Sample
    #     samples = torch.empty(0, device = self.device)
    #     while samples.shape[0] < num_samples:

    #         # Get token for sample indices
    #         token = data.get_sample_tensor(
    #             sample_indices,
    #             timestamp_encodings=timestamp_encodings
    #         )

    #         # Repeat token to give correct batch size
    #         tokens = token.unsqueeze(0).repeat(batch_size,1,1,1)

    #         #if self.device == torch.device("mps"):
    #         tokens = tokens.to(self.device)

    #         # Make sure we do not make too many predictions if num_samples % batch_size is not equal to 0
    #         if num_samples - samples.shape[0] < batch_size:
    #             tokens = tokens[:num_samples - samples.shape[0]]

    #         if self.scaling:
    #             tokens = self.scaler(tokens)

    #         # Sample initial white noise
    #         tokens = self.diffuser.denoise(
    #             tokens = tokens,
    #             epsilon = None,
    #             n = self.diff_steps
    #         )

    #         # Compute each diffusion step
    #         for n in range(self.diff_steps,0,-1):

    #             # Calculate predicted noise
    #             epsilon = self.forward(torch.clone(tokens))

    #             # Calculate x_n
    #             tokens = self.diffuser.denoise(
    #                 tokens  = tokens,
    #                 epsilon = epsilon,
    #                 n = n
    #             )


    #         if self.scaling:
    #             tokens = self.scaler.unscale(tokens)

    #         # Add denoised tokens to samples
    #         samples = torch.cat((samples, torch.clone(tokens[:,:,-self.prediction_length:,0])))

    #     return samples

