# Library imports
import torch
import math
import time
import copy
import numpy as np
import calendar


# Module imports
from torch import nn, Tensor, optim
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Callable, Optional, Dict, Tuple
from pandas import DataFrame
from data import TimeFusionDataset
from diffusion import Diffuser
from pandas import DatetimeIndex
from torch.utils.data import DataLoader

### The following segment of code is from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, restore_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.restore_weights = restore_weights
        self.best_weights = None

    def early_stop(self, validation_loss, model):
        if validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
        else:
            self.counter = 0

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            if self.restore_weights:
                self.best_weights = copy.deepcopy(model.state_dict())

        if self.counter >= self.patience:
            if self.restore_weights:
                return True, self.best_weights
            return True, None
        
        return False, None
###

class MeanScaler(nn.Module):
    def __init__(
            self, 
            context_length: int,
            min_scale: float,
            device: torch.device
        ):

        # Init nn.Module base class
        super().__init__()

        self.context_length = context_length
        self.min_scale = min_scale
        self.device = device
        self.scales = None

    def forward(self, x: Tensor):
        """
        Args:
            x: Tensor of shape (batch size, timeseries_dim, total length, datapoint_dim) or (timeseries_dim, total length, datapoint_dim))
        """
        # Scale data
        scales = torch.maximum(torch.mean(x[:,:,:self.context_length,0],dim=2),torch.full(x[:,:,0,0].shape,self.min_scale,device=self.device))
        x[:,:,:,0] /= scales.view(x.shape[:2]+ tuple([1]))

        self.scales = scales

        return x
    
    def unscale(self, x: Tensor):
        x[:,:,:,0] *= self.scales.view(x.shape[:2]+ tuple([1]))
        return x


# Class accessible to end-user
class TimeFusion(nn.Module):

    def __init__(
            self,
            context_length: int,
            prediction_length: int,
            timeseries_shape: Tuple[int],
            num_encoder_layers: int = 6,
            scaling: bool = False,
            d_model: int = 32, 
            nhead: int = 8,
            dim_feedforward: int = 2048,
            diff_steps: int = 100,
            betas: List[float] = None,
            device: torch.device = torch.device("cpu"),
        ) -> None:
        
        """
        Args:
            context_length: The number of historical datapoints given to the network each time it makes a prediction.
            prediction_length: The number of datapoints into the future which the network will predict.
            timeseries_shape: The shape of the time-series at each time-instant (timeseries_dim, datapoint_dim).
            num_encoder_layers: The number of encoder layers which in the Transformer part of the network.
            d_model: The dimension of the tokens used by the Transformer part of the network.
            nhead: The number of heads in the attention layers of the Transformer part of the network.
            dim_feedforward: The dimension of the feedforward network model in Transformer.
            diff_steps: Number of diffusion steps.
            betas: Betas values for diffusion, will usually be an ascending list of numbers.
            device: The device on which computations should be performed (e.g. "cpu" or "cuda0").
        """
        
        # Init nn.Module base class
        super().__init__()

        ### Set instance variables ###
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.total_length  = context_length + prediction_length
        self.timeseries_shape = timeseries_shape
        self.scaling = scaling
        self.diff_steps = diff_steps
        self.device = device


        ### Initialize all components of the network ###
        self.diffuser = Diffuser(
            prediction_length = prediction_length,
            diff_steps = diff_steps,
            betas = betas,
            device = device
        )

        if scaling:
            self.scaler = MeanScaler(context_length,0.01,device=device)

        self.embedding = nn.Linear(
            in_features = timeseries_shape[0]*timeseries_shape[1], 
            out_features = d_model,
            device = device
        )

        self.positional_encoding = PositionalEncoding(
           d_model = d_model,
           dropout = 0,
           device = device
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model = d_model, 
                nhead = nhead,
                dim_feedforward = dim_feedforward,
                dropout = 0,
                batch_first = True,
                device = device
            ),
            num_layers = num_encoder_layers
        )
        
        self.linear = nn.Linear(
            in_features = d_model,
            out_features = timeseries_shape[0],
            device = device
        )
    
    def forward(self, x: Tensor):
        """
        Args:
            x: Tensor of shape (batch size, timeseries_dim, total length, datapoint_dim) or (timeseries_dim, total length, datapoint_dim))
        """

        # If data is unbatched, add a batch dimension
        if x.dim() == 3:
            x.unsqueeze(0)

        # Pass input through network
        x = torch.permute(x, (0, 2, 1, 3))
        x = torch.flatten(x, start_dim=2)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.linear(x[:,self.context_length:])
        x = torch.permute(x, (0, 2, 1))

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
            for i, tokens in enumerate(train_loader, start = 1):
                #if self.device == torch.device("mps"):
                tokens = tokens.to(self.device)

                if self.scaling:
                    tokens = self.scaler(tokens)

                # Diffuse data
                tokens, targets = self.diffuser.diffuse(tokens)

                # Zero gradients
                optimizer.zero_grad()

                # Forward, loss calculation, backward, optimizer step
                predictions = self.forward(tokens)
                loss = loss_function(predictions,targets)
                loss.backward()
                optimizer.step()

                # Print training statistics
                running_loss += loss.item()
                average_loss = running_loss / i
                stat_string = "|" + "="*(30*i // len(train_loader)) + " "*(30 - (30*i // len(train_loader))) + f"|  Batch: {i} / {len(train_loader)}, Epoch: {epoch} / {epochs}, Average Loss: {average_loss:.4f}"
                print("\u007F"*512,stat_string,end="\r")

            if lr_scheduler:
                lr_scheduler.step()

            if val_loader is not None:
                with torch.no_grad():
                    running_loss = {key:0 for key in val_metrics.keys()}
                    for tokens in val_loader:
                        #if self.device == torch.device("mps"):
                        tokens = tokens.to(self.device)

                        if self.scaling:
                            tokens = self.scaler(tokens)

                        # Diffuse data
                        tokens, targets = self.diffuser.diffuse(tokens)

                        # Calculate prediction metrics
                        predictions = self.forward(tokens)
                        for key, metric_func in val_metrics.items():
                            running_loss[key] += metric_func(predictions,targets).item() 
                    
                    for metric, value in running_loss.items():
                        stat_string += f", {metric}: {value / len(val_loader):.4f}"

                    print("\u007F"*512,stat_string)

                    if not early_stopper is None:
                        stop, weights = early_stopper.early_stop(running_loss["val_loss"] / len(val_loader),self)
                        if stop:
                            if not weights is None:
                                self.load_state_dict(weights)
                            break
            else:
                if not early_stopper is None:
                        stop, weights = early_stopper.early_stop(average_loss,self)
                        if stop:
                            if not weights is None:
                                self.load_state_dict(weights)
                            break

                # New line for printing statistics
                print()

        if (not early_stopper is None) and early_stopper.restore_weights:
            self.load_state_dict(early_stopper.best_weights)


    @torch.no_grad()
    def sample(
        self,
        data: TimeFusionDataset,
        sample_indices: List[DatetimeIndex],
        num_samples: int = 1,
        batch_size: int = 64,
        timestamp_encodings: List[Callable] = [
            lambda x: math.sin(2*math.pi*x.hour / 24),
            lambda x: math.sin(2*math.pi*x.weekday() / 7),
            lambda x: math.sin(2*math.pi*x.day / calendar.monthrange(x.year, x.month)[1]),
            lambda x: math.sin(2*math.pi*x.month / 12),
            lambda x: math.cos(2*math.pi*x.hour / 24),
            lambda x: math.cos(2*math.pi*x.weekday() / 7),
            lambda x: math.cos(2*math.pi*x.day / calendar.monthrange(x.year, x.month)[1]),
            lambda x: math.cos(2*math.pi*x.month / 12),
        ],
    ):
        
        """
        Args:
            data: Historical data which is fed as context to the network.
            timestamps: Indexes at which to predict future values, length should be prediction length.
            num_samples: Number of samples to sample from the network.
            batch_size: The number of samples to process at the same time.

        Note:
            1. For optimal performance, num_samples should be divisible by batch size.
            2. The timestamps must be drawn from the same distribution as those from the training data for best performance
        """

        # Set the network into evaluation mode
        self.train(False)
        

        # Sample
        samples = torch.empty(0, device = self.device)
        while samples.shape[0] < num_samples:

            # Get token for sample indices
            token = data.get_sample_tensor(
                sample_indices,
                timestamp_encodings=timestamp_encodings
            )

            # Repeat token to give correct batch size
            tokens = token.unsqueeze(0).repeat(batch_size,1,1,1)

            #if self.device == torch.device("mps"):
            tokens = tokens.to(self.device)

            # Make sure we do not make too many predictions if num_samples % batch_size is not equal to 0
            if num_samples - samples.shape[0] < batch_size:
                tokens = tokens[:num_samples - samples.shape[0]]

            if self.scaling:
                tokens = self.scaler(tokens)

            # Sample initial white noise
            tokens = self.diffuser.denoise(
                tokens = tokens,
                epsilon = None,
                n = self.diff_steps
            )

            # Compute each diffusion step
            for n in range(self.diff_steps,0,-1):

                # Calculate predicted noise
                epsilon = self.forward(torch.clone(tokens))

                # Calculate x_n
                tokens = self.diffuser.denoise(
                    tokens  = tokens,
                    epsilon = epsilon,
                    n = n
                )


            if self.scaling:
                tokens = self.scaler.unscale(tokens)
                
            # Add denoised tokens to samples
            samples = torch.cat((samples, torch.clone(tokens[:,:,-self.prediction_length:,0])))

        return samples


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, device,dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len,device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_length, embedding_dim]
        """
        x += self.pe[:x.size(1)]
        return self.dropout(x)
    


# TODO:
# 5. Definitely need to speed up Transformers, must reduce input length or use a more efficient attention mechanism
# 6. Weight decay?
# 10. Add time to training statistics
