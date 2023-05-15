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

import math

import torch
from torch import nn
import torch.nn.functional as F


class DiffusionEmbedding(nn.Module):
    def __init__(self, dim, proj_dim, max_steps=500):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(dim, max_steps), persistent=False
        )
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, dim, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(dim).unsqueeze(0)  # [1,dim]
        table = steps * 10.0 ** (dims * 4.0 / dim)  # [T,dim]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )
        self.diffusion_projection = nn.Linear(hidden_size, residual_channels)
        self.conditioner_projection = nn.Conv1d(
            1, 2 * residual_channels, 1, padding=2, padding_mode="circular"
        )
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)

        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        y = F.leaky_relu(y, 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class CondUpsampler(nn.Module):
    def __init__(self, cond_length, target_dim):
        super().__init__()
        self.linear1 = nn.Linear(cond_length, target_dim // 2)
        self.linear2 = nn.Linear(target_dim // 2, target_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.linear2(x)
        x = F.leaky_relu(x, 0.4)
        return x


class EpsilonTheta(nn.Module):
    def __init__(
        self,
        target_dim,
        cond_length,
        time_emb_dim=16,
        residual_layers=8,
        residual_channels=8,
        dilation_cycle_length=2,
        residual_hidden=64,
    ):
        super().__init__()
        self.input_projection = nn.Conv1d(
            1, residual_channels, 1, padding=2, padding_mode="circular"
        )
        self.diffusion_embedding = DiffusionEmbedding(
            time_emb_dim, proj_dim=residual_hidden
        )
        self.cond_upsampler = CondUpsampler(
            target_dim=target_dim, cond_length=cond_length
        )
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels=residual_channels,
                    dilation=2 ** (i % dilation_cycle_length),
                    hidden_size=residual_hidden,
                )
                for i in range(residual_layers)
            ]
        )
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 3)
        self.output_projection = nn.Conv1d(residual_channels, 1, 3)

        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, inputs, time, cond):
        x = self.input_projection(inputs)
        x = F.leaky_relu(x, 0.4)

        diffusion_step = self.diffusion_embedding(time)
        cond_up = self.cond_upsampler(cond)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_up, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.leaky_relu(x, 0.4)
        x = self.output_projection(x)
        return x

# class DiffusionEmbedding(nn.Module):

#     def __init__(self, dim: int, proj_dim: int, device: torch.device, max_steps: int = 500):
#         super().__init__()

#         # Look up table for sine encodings
#         step = torch.arange(max_steps, device = device).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
#         pe = torch.zeros(max_steps, dim, device = device)
#         pe[:, 0::2] = torch.sin(step * div_term)
#         pe[:, 1::2] = torch.cos(step * div_term)
#         self.register_buffer('pe', pe)

#         # FC network
#         self.projection1 = nn.Linear(dim, proj_dim)
#         self.tanh1 = nn.Tanh()
#         self.projection2 = nn.Linear(proj_dim, proj_dim)

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Args:
#             x: Tensor with shape [batch_size] giving the diffusion step
#         """
#         x = self.pe[x]
#         x = self.projection1(x)
#         x = self.tanh1(x)
#         x = self.projection2(x)
#         return x
    


# # Class accessible to end-user
# class TimeFusion(nn.Module):

#     def __init__(
#             self,
#             context_length: int,
#             prediction_length: int,
#             input_size: int,
#             num_ts: int,
#             num_recurrent_layers: int = 2,
#             scaling: bool = False,
#             diff_steps: int = 100,
#             betas: List[float] = None,
#             device: torch.device = torch.device("cpu"),
#         ) -> None:
        
#         """
#         Args:
#             context_length: The number of historical datapoints given to the network each time it makes a prediction.
#             prediction_length: The number of datapoints into the future which the network will predict.
#             timeseries_shape: The shape of the time-series at each time-instant (timeseries_dim, datapoint_dim).
#             num_encoder_layers: The number of encoder layers which in the Transformer part of the network.
#             d_model: The dimension of the tokens used by the Transformer part of the network.
#             nhead: The number of heads in the attention layers of the Transformer part of the network.
#             dim_feedforward: The dimension of the feedforward network model in Transformer.
#             diff_steps: Number of diffusion steps.
#             betas: Betas values for diffusion, will usually be an ascending list of numbers.
#             device: The device on which computations should be performed (e.g. "cpu" or "cuda0").
#         """
        
#         # Init nn.Module base class
#         super().__init__()

#         ### Set instance variables ###
#         self.context_length = context_length
#         self.prediction_length = prediction_length
#         self.total_length  = context_length + prediction_length
#         self.scaling = scaling
#         self.diff_steps = diff_steps
#         self.input_size = input_size
#         self.num_ts = num_ts
#         self.num_recurrent_layers = num_recurrent_layers
#         self.device = device

#         ### Initialize all components of the network ###
#         if scaling:
#             self.scaler = MeanScaler(
#                 device = device,
#                 min_scale = 0.01
#             )

#         self.diffuser = Diffuser(
#             diff_steps = diff_steps,
#             betas = betas,
#             device = device
#         )

#         self.diff_embedding = DiffusionEmbedding(
#             dim = 32,
#             proj_dim = 32,
#             device = device
#         )
        
#         self.recurrent = nn.LSTM(
#             input_size = 20,
#             hidden_size = 40,
#             num_layers = self.num_recurrent_layers,
#             batch_first = True
#         )
        
#         self.linear1 = nn.Linear(
#             in_features = 92,
#             out_features = 92,
#             device = device
#         )

#         self.linear2 = nn.Linear(
#             in_features = 92,
#             out_features = 20,
#             device = device
#         )

#         self.relu = nn.ReLU()
    
#     def forward(self, context: Tensor, x: Tensor, n: Tensor):
#         """
#         Args:
#             context: Tensor of shape (batch size, time-series dim, context length)
#             x: Tensor of shape (batch size, time-series dim)
#             n: Tensor of shape (batch size)
#         """
#         context = context.permute((0,2,1))
#         n = self.diff_embedding(n)
#         #n = n / 100.0
#         h, _  = self.recurrent(context)
#         x = torch.concat([h[:,-1],x,n],dim=1)

#         x = self.linear1(x)
#         x = self.relu(x)
#         x = self.linear2(x)

#         return x

#     # Function to train TimeFusion network
#     def train_network(self,
#             train_loader: DataLoader, 
#             epochs: int,
#             val_loader: Optional[DataLoader] = None,
#             val_metrics: Optional[Dict[str,Callable]] = None,
#             loss_function: Callable = nn.MSELoss(),
#             optimizer: optim.Optimizer = None,
#             lr_scheduler: _LRScheduler = None,
#             early_stopper: EarlyStopper = None,
#         ) -> None:
#         """
#         Args:
#             train_loader: Generator which provides batched training data.
#             epochs: The number of epochs to train for.
#             train_loader: Generator which provides batched validation data.
#             val_metrics: Name of metrics and callable functions to calculate them which measure the performance of the network.
#             loss_function: Function to measure how well predictions match targets.
#             optimizer: Optimizer used to update weights.
#             lr_scheduler: Learning rate scheduler which modifies learning rate after each epoch.
#         """

#         # Set the network into training mode
#         self.train(True)

#         # Set default optimizer
#         if optimizer is None:
#             optimizer = optim.Adam(params = self.parameters(), lr = 1e-4)

#         # Set default validation metrics
#         val_metrics = val_metrics | {"val_loss": loss_function}
            

#         for epoch in range(1, epochs + 1):

#             running_loss = 0
#             for i, data in enumerate(train_loader, start = 1):
#                 context, target = data
#                 #if self.device == torch.device("mps"):
#                 context = context.to(self.device)
#                 target = target.to(self.device)

#                 if self.scaling:
#                     context = self.scaler(context)

#                 target /= self.scaler.scales


#                 # Diffuse data
#                 x, target, n = self.diffuser.diffuse(target)

#                 # Zero gradients
#                 optimizer.zero_grad()

#                 # Forward, loss calculation, backward, optimizer step
#                 predictions = self.forward(context, x, n)
#                 #predictions = self.forward(context, None, None)
#                 loss = loss_function(predictions,target)
#                 loss.backward()
#                 optimizer.step()

#                 # Print training statistics
#                 running_loss += loss.item()
#                 average_loss = running_loss / i
#                 stat_string = "|" + "="*(30*i // len(train_loader)) + " "*(30 - (30*i // len(train_loader))) + f"|  Batch: {i} / {len(train_loader)}, Epoch: {epoch} / {epochs}, Average Loss: {average_loss:.4f}"
#                 print("\u007F"*512,stat_string,end="\r")

#             if lr_scheduler:
#                 lr_scheduler.step()

#             # if val_loader is not None:
#             #     with torch.no_grad():
#             #         running_loss = {key:0 for key in val_metrics.keys()}
#             #         for tokens in val_loader:
#             #             #if self.device == torch.device("mps"):
#             #             tokens = tokens.to(self.device)

#             #             if self.scaling:
#             #                 tokens = self.scaler(tokens)

#             #             # Diffuse data
#             #             tokens, targets = self.diffuser.diffuse(tokens)

#             #             # Calculate prediction metrics
#             #             predictions = self.forward(tokens)
#             #             for key, metric_func in val_metrics.items():
#             #                 running_loss[key] += metric_func(predictions,targets).item() 
                    
#             #         for metric, value in running_loss.items():
#             #             stat_string += f", {metric}: {value / len(val_loader):.4f}"

#             #         print("\u007F"*512,stat_string)

#             #         if not early_stopper is None:
#             #             stop, weights = early_stopper.early_stop(running_loss["val_loss"] / len(val_loader),self)
#             #             if stop:
#             #                 if not weights is None:
#             #                     self.load_state_dict(weights)
#             #                 break
#             # else:
#             #     if not early_stopper is None:
#             #             stop, weights = early_stopper.early_stop(average_loss,self)
#             #             if stop:
#             #                 if not weights is None:
#             #                     self.load_state_dict(weights)
#             #                 break

#                 # New line for printing statistics
#                 print()

#         # if (not early_stopper is None) and early_stopper.restore_weights:
#         #     self.load_state_dict(early_stopper.best_weights)


#     # @torch.no_grad()
#     # def sample(
#     #     self,
#     #     data: TimeFusionDataset,
#     #     sample_indices: List[DatetimeIndex],
#     #     num_samples: int = 1,
#     #     batch_size: int = 64,
#     #     timestamp_encodings: List[Callable] = [
#     #         lambda x: math.sin(2*math.pi*x.hour / 24),
#     #         lambda x: math.sin(2*math.pi*x.weekday() / 7),
#     #         lambda x: math.sin(2*math.pi*x.day / calendar.monthrange(x.year, x.month)[1]),
#     #         lambda x: math.sin(2*math.pi*x.month / 12),
#     #         lambda x: math.cos(2*math.pi*x.hour / 24),
#     #         lambda x: math.cos(2*math.pi*x.weekday() / 7),
#     #         lambda x: math.cos(2*math.pi*x.day / calendar.monthrange(x.year, x.month)[1]),
#     #         lambda x: math.cos(2*math.pi*x.month / 12),
#     #     ],
#     # ):
        
#     #     """
#     #     Args:
#     #         data: Historical data which is fed as context to the network.
#     #         timestamps: Indexes at which to predict future values, length should be prediction length.
#     #         num_samples: Number of samples to sample from the network.
#     #         batch_size: The number of samples to process at the same time.

#     #     Note:
#     #         1. For optimal performance, num_samples should be divisible by batch size.
#     #         2. The timestamps must be drawn from the same distribution as those from the training data for best performance
#     #     """

#     #     # Set the network into evaluation mode
#     #     self.train(False)
        

#     #     # Sample
#     #     samples = torch.empty(0, device = self.device)
#     #     while samples.shape[0] < num_samples:

#     #         # Get token for sample indices
#     #         token = data.get_sample_tensor(
#     #             sample_indices,
#     #             timestamp_encodings=timestamp_encodings
#     #         )

#     #         # Repeat token to give correct batch size
#     #         tokens = token.unsqueeze(0).repeat(batch_size,1,1,1)

#     #         #if self.device == torch.device("mps"):
#     #         tokens = tokens.to(self.device)

#     #         # Make sure we do not make too many predictions if num_samples % batch_size is not equal to 0
#     #         if num_samples - samples.shape[0] < batch_size:
#     #             tokens = tokens[:num_samples - samples.shape[0]]

#     #         if self.scaling:
#     #             tokens = self.scaler(tokens)

#     #         # Sample initial white noise
#     #         tokens = self.diffuser.denoise(
#     #             tokens = tokens,
#     #             epsilon = None,
#     #             n = self.diff_steps
#     #         )

#     #         # Compute each diffusion step
#     #         for n in range(self.diff_steps,0,-1):

#     #             # Calculate predicted noise
#     #             epsilon = self.forward(torch.clone(tokens))

#     #             # Calculate x_n
#     #             tokens = self.diffuser.denoise(
#     #                 tokens  = tokens,
#     #                 epsilon = epsilon,
#     #                 n = n
#     #             )


#     #         if self.scaling:
#     #             tokens = self.scaler.unscale(tokens)

#     #         # Add denoised tokens to samples
#     #         samples = torch.cat((samples, torch.clone(tokens[:,:,-self.prediction_length:,0])))

#     #     return samples

