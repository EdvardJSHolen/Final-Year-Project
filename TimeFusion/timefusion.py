# Library imports
import torch
import math
import pandas as pd
import numpy as np

import time

# Module imports
from torch import nn, Tensor, optim
from typing import List, Callable, Optional, Dict
from datetime import timedelta
from math import pi
from pandas import DataFrame
from diffusion import BatchLoader

# Class accessible to end-user
class TimeFusion(nn.Module):

    def __init__(
            self,
            datapoint_dim: int,
            time_series_dim: int,
            context_length: int,
            prediction_length: int,
            start_length: int = 0,
            indices: List[int] = [], 
            timestamps: List[int] = [],
            d_model: int = 32, 
            nhead: int = 8, 
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6, 
            dim_feedforward: int = 128,
            diff_steps: int = 100,
            betas: List[float] = None,
            device: torch.device = torch.device("cpu"),
        ) -> None:
        
        """
        Args:
            datapoint_dim: The number of elements in a single datapoint
            indices: Elements of datapoints to be encoded with sine/cos waves of arbitrary frequency
            timestamps: Elements of datapoints containing timestamps to be encoded with sine/cos waves.
            d_model: The number of features in the encoder/decoder inputs
            nhead: The number of heads in the multi-head attention
            num_encoder_layers: The number of sub-encoder-layers in the encoder
            num_decoder_layers: The number of sub-decoder-layers in the decoder
            dim_feedforward: The dimension of the feedforward network model in Transformer
            device: The device on which computations should be performed (e.g. "cpu" or "cuda0")
        """
        
        # Init nn.Module base class
        super().__init__()

        # Set instance variables
        self.datapoint_dim = datapoint_dim
        self.time_series_dim = time_series_dim
        self.total_length  = context_length + prediction_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.start_length = start_length
        self.device = device
        self.diff_steps = diff_steps
    
        if betas is None:
            self.betas = np.linspace(1e-4, 0.1, diff_steps)

        self.alphas = 1.0 - self.betas
        self.bar_alphas = np.cumprod(self.alphas)
        
        self.embedding = nn.Linear(datapoint_dim*time_series_dim, d_model,device=device)

        self.positional_encoding = PositionalEncoding(
           d_model=d_model,
           dropout=0,
           max_len=500,
           device=device
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead,
                dropout=0,
                batch_first=True,
                device=device
            ),
            num_layers=num_encoder_layers
        )
        
        self.linear = nn.Linear((context_length + prediction_length) * d_model, prediction_length*time_series_dim,device=device)
    
    def forward(self, x):
        x_original = x
        x = torch.permute(x, (0, 2, 1, 3))
        x = torch.flatten(x, start_dim=2)
        x = self.embedding(x)
        x = torch.reshape(x, (x_original.shape[0],self.total_length,-1))
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = torch.reshape(x,(x_original.shape[0],-1,self.time_series_dim))
        x = torch.permute(x, (0, 2, 1))
        return x

    # Function to train TimeFusion network
    def train(self,
            train_data: DataFrame, 
            epochs: int,
            batch_size: int = 64,
            optimizer: optim.Optimizer = None,
            loss_function: Callable = nn.MSELoss(),
            val_data: Optional[DataFrame] = None,
            val_metrics: Optional[Dict[str,Callable]] = None
        ) -> None:
        """
        Args:
            data: Pandas DataFrame with timestamps as the index and a column for each time-series. Cells should be filled with
            nan-values when a time-series does not have a datapoint at a given timestamp.
            epochs: The number of epochs to train for.
            num_batches_per_epoch: The number of batches to train on in a given epoch.
            batch_size: The number of samples to process at the same time.
            optimizer: Optimizer used to train weights
            loss_function: Function to measure how well predictions match targets.
        """

        # Set default optimizer
        if optimizer is None:
            optimizer = optim.Adam(params=self.parameters(),lr=1e-4)

        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=epochs)


        train_loader = BatchLoader(
            data = train_data,
            batch_size = batch_size,
            context_length = self.context_length,
            prediction_length = self.prediction_length,
            diff_steps = self.diff_steps,
            device = self.device
        )

        if not val_data is None:
            if val_metrics is None:
                val_metrics = {"Val loss": loss_function}
            
            val_loader = BatchLoader(
                data = val_data,
                batch_size = batch_size,
                context_length = self.context_length,
                prediction_length = self.prediction_length,
                diff_steps = self.diff_steps,
                device = self.device
            )
        

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

            scheduler.step()

            if val_data is not None:
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
        # TODO: MAKE THIS WORK WITH IRREGULAR TIME-SERIES DATA
        # TODO: Optimize by splitting encoder from rest of transformer
        # For optimal performance, num_samples should be divisible by batch size
        # Does not yet work when start length is longer than context length

        # Generate context Tensor
        context = torch.empty(0, device = self.device)
        for j, column in enumerate(data.columns):
            col_values = data[column].dropna()
            col_tensor = torch.tensor(
                [
                    list(col_values.iloc[-self.context_length:]), # Value
                    list(np.array(col_values.index[-self.context_length:] - data.index[-1])/self.context_length), # Timestamp
                    #list(range(self.context_length)), # Datapoint index
                    #list(np.full(shape=(context_length),fill_value=j)), # Time-series index
                    list(np.zeros(shape=(self.context_length))) # Diffusion index
                ],
                dtype = torch.float32,
                device = self.device
            ).t()
            context = torch.cat((context, col_tensor[None,:]))

        # Create query Tensor (non-diffused)
        query = torch.empty(0, device = self.device)
        for j, column in enumerate(data.columns):
            col_tensor = torch.tensor(
                [
                    [0]*self.prediction_length, # Value
                    list(np.array(timestamps[j] - data.index[-1])/self.context_length), # Timestamp
                    #list(range(self.context_length,self.prediction_length + self.context_length)), # Datapoint index
                    #list(np.full(shape=(prediction_length + start_length),fill_value=j)), # Time-series index
                    list(np.zeros(shape=(self.prediction_length))) # Diffusion index
                ],
                dtype = torch.float32,
                device = self.device
            ).t()
            query = torch.cat((query, col_tensor[None,:]))

        # Combine queries and context
        tokens = torch.cat((context,query),dim=-2)

        # Copy context and query tensor to be of same size as batch size (HACK)
        # context = context.unsqueeze(0).repeat([batch_size] + [1]*context.dim())
        # query = query.unsqueeze(0).repeat([batch_size] + [1]*query.dim())
        tokens = tokens.unsqueeze(0).repeat([batch_size] + [1]*tokens.dim())

        # Sample
        samples = torch.empty(0, device = self.device)
        # iterations = torch.empty(0, device = self.device)
        # iterations2 = torch.empty(0, device = self.device)
        while samples.shape[0] < num_samples:

            # Make sure we do not make too many predictions if num_samples % batch_size is not equal to 0
            if num_samples - samples.shape[0] < batch_size:
                tokens = tokens[:num_samples - samples.shape[0]]


            # Sample initial white noise
            tokens[:,:,self.context_length:,0] = torch.empty(size=tokens[:,:,self.context_length:,0].shape,device=self.device).normal_()

            # Compute each diffusion step
            for n in range(self.diff_steps,0,-1):

                # Set diffusion step
                #tokens[:,:,self.context_length:,-1] = 2*n / self.diff_steps - 1
                tokens[:,:,self.context_length:,-1] = torch.full(tokens[:,:,self.context_length:,-1].shape, 2*n / self.diff_steps - 1)

                # Calculate output
                output = self.forward(tokens)

                #iterations2 = torch.cat((iterations2, output))

                # Give new value to query
                if n > 1:
                    z = torch.empty(size=tokens[:,:,self.context_length:,0].shape,device=self.device).normal_()
                else: 
                    z = torch.zeros(size=tokens[:,:,self.context_length:,0].shape,device=self.device)

                tokens[:,:,self.context_length:,0] = (1/math.sqrt(self.alphas[n-1]))*(tokens[:,:,self.context_length:,0] - (self.betas[n-1]/math.sqrt(1-self.bar_alphas[n-1]))*output) + math.sqrt(self.betas[n-1])*z

            # Add denoised queries to samples
            #samples = torch.cat((samples, query[:,:,self.start_length:]))
            samples = torch.cat((samples, tokens[:,:,self.context_length:,0]))
        return samples#, iterations, iterations2


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, device,dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len,device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2,device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model,device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, indices = None) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_length, embedding_dim]
        """
        x += self.pe[:x.size(1)]
        return self.dropout(x)


# TODO:
# 1. Investigate where it is beneficial to add @torch.no_grad() decorator
# 2. Investigate weight initilization for linear layers
# 3. Split Transformer into encoder and decoder such that I can reduce computation during inference
# 4. Normalize input data?
# 5. Definitely need to speed up Transformers, must reduce input length or use a more efficient attention mechanism
# 6. Weight decay?
# 7. Use learning rate scheduler
# 8. Figure out how to define "epoch" for time-series data
# 9. Refine method of feeding data to network such that we can also feed covariates. Need to think how to represent it in pandas dataframe. Maybe just use tuple entries. Can use multiindex
# 10. Add time to training statistics
# 11. Need to consolidate beta schedules, currently define them two places.
# 12. NEED TO SET model.eval() ????



# NOTE:
# 1. Definitiely should use @torch.no_grad() when measuring inference as this avoids the tracking of gradients, making it 6! times faster