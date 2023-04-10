# Library imports
import torch
import math
import time
import numpy as np


# Module imports
from torch import nn, Tensor, optim
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Callable, Optional, Dict, Tuple
from pandas import DataFrame
from diffusion import BatchLoader

# Class accessible to end-user
class TimeFusion(nn.Module):

    def __init__(
            self,
            context_length: int,
            prediction_length: int,
            timeseries_shape: Tuple[int],
            num_encoder_layers: int = 6,
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

        # Network parameters
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.total_length  = context_length + prediction_length
        self.timeseries_shape = timeseries_shape

        # Diffusion parameters
        self.diff_steps = diff_steps
        self.betas = betas

        if betas is None:
            self.betas = np.linspace(1e-4, 0.1, diff_steps)

        self.alphas = 1.0 - self.betas
        self.bar_alphas = np.cumprod(self.alphas)

        # Training parameters
        self.device = device


        ### Initialize all components of the network ###

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
            train_loader: BatchLoader, 
            epochs: int,
            val_loader: Optional[BatchLoader] = None,
            val_metrics: Optional[Dict[str,Callable]] = None,
            loss_function: Callable = nn.MSELoss(),
            optimizer: optim.Optimizer = None,
            lr_scheduler: _LRScheduler = None
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
        if val_metrics is None:
            val_metrics = {"Val loss": loss_function}
            

        for epoch in range(1, epochs + 1):

            running_loss = 0
            for i, batch in enumerate(train_loader, start = 1):

                # Split data into context, queries and prediction targets
                tokens, targets = batch

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
                stat_string = "|" + "="*(30*i // train_loader.num_batches) + " "*(30 - (30*i // train_loader.num_batches)) + f"|  Batch: {i} / {train_loader.num_batches}, Epoch: {epoch} / {epochs}, Average Loss: {average_loss:.4f}"
                print("\u007F"*512,stat_string,end="\r")

            if lr_scheduler:
                lr_scheduler.step()

            if val_loader is not None:
                with torch.no_grad():
                    running_loss = {key:0 for key in val_metrics.keys()}
                    for i, batch in enumerate(val_loader, start = 1):
                        tokens, targets = batch
                        predictions = self.forward(tokens)
                        for key, metric_func in val_metrics.items():
                            running_loss[key] += metric_func(predictions,targets).item()
                    
                    for metric, value in running_loss.items():
                        stat_string += f", {metric}: {value / val_loader.num_batches:.4f}"

                    print("\u007F"*512,stat_string)
            else:
                # New line for printing statistics
                print()


    @torch.no_grad()
    def sample(
        self,
        data: DataFrame,
        timestamps: List[float],
        num_samples: int = 1,
        batch_size: int = 32
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

        # Generate context Tensor
        context = torch.empty(0, device = self.device)
        for column in data.columns:
            col_values = data[column].dropna()
            col_tensor = torch.tensor(
                [
                    list(col_values.iloc[-self.context_length:]), # Value
                    list(np.array(col_values.index[-self.context_length:] - data.index[-1])/self.context_length), # Timestamp
                    list(np.zeros(shape=(self.context_length))) # Diffusion index
                ],
                dtype = torch.float32,
                device = self.device
            ).t()
            context = torch.cat((context, col_tensor.unsqueeze(0)))

        # Create query Tensor (non-diffused)
        query = torch.empty(0, device = self.device)
        for j, column in enumerate(data.columns):
            col_tensor = torch.tensor(
                [
                    [0]*self.prediction_length, # Value
                    list(np.array(timestamps[j] - data.index[-1])/self.context_length), # Timestamp
                    list(np.zeros(shape=(self.prediction_length))) # Diffusion index
                ],
                dtype = torch.float32,
                device = self.device
            ).t()
            query = torch.cat((query, col_tensor.unsqueeze(0)))

        # Combine queries and context
        tokens = torch.cat((context,query), dim= -2)

        # Repeat tokens such that the first dimension of the tokens tensor is equal to the batch size
        tokens = tokens.unsqueeze(0).repeat([batch_size] + [1]*tokens.dim())

        # Sample
        samples = torch.empty(0, device = self.device)
        while samples.shape[0] < num_samples:

            # Make sure we do not make too many predictions if num_samples % batch_size is not equal to 0
            if num_samples - samples.shape[0] < batch_size:
                tokens = tokens[:num_samples - samples.shape[0]]

            # Sample initial white noise
            tokens[:,:,self.context_length:,0] = torch.empty(size=tokens[:,:,self.context_length:,0].shape,device=self.device).normal_()

            # Compute each diffusion step
            for n in range(self.diff_steps,0,-1):

                # Set diffusion step
                tokens[:,:,self.context_length:,-1] = torch.full(tokens[:,:,self.context_length:,-1].shape, 2*n / self.diff_steps - 1)

                # Calculate output
                output = self.forward(tokens)

                # Give new value to query
                if n > 1:
                    z = torch.empty(size=tokens[:,:,self.context_length:,0].shape,device=self.device).normal_()
                else: 
                    z = torch.zeros(size=tokens[:,:,self.context_length:,0].shape,device=self.device)

                tokens[:,:,self.context_length:,0] = (1/math.sqrt(self.alphas[n-1]))*(tokens[:,:,self.context_length:,0] - (self.betas[n-1]/math.sqrt(1-self.bar_alphas[n-1]))*output) + math.sqrt(self.betas[n-1])*z

            # Add denoised queries to samples
            samples = torch.cat((samples, tokens[:,:,self.context_length:,0]))

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
# 11. Need to consolidate beta schedules, currently define them two places.
# 12. NEED TO SET model.eval() ????
