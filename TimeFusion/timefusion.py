# Library imports
import torch
import pandas as pd

# Module imports
from torch import nn, Tensor, optim
from typing import List, Callable
from datetime import timedelta
from math import pi
from pandas import DataFrame
from diffusion import batch_loader

# Class accessible to end-user
class TimeFusion(nn.Module):

    def __init__(
            self,
            datapoint_dim: int,
            context_length: int,
            prediction_length: int,
            indices: List[int] = [], 
            timestamps: List[int] = [],
            d_model: int = 32, 
            nhead: int = 8, 
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6, 
            dim_feedforward: int = 128,
            diff_steps: int = 100,
            device: torch.device = torch.device("cpu"),
        ):
        
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
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.device = device
        self.diff_steps = diff_steps

        self.encoding = PositionalEncoding(
            datapoint_dim = datapoint_dim,
            indices = indices,
            timestamps = timestamps,
            device = device,
        )

        self.embedding = Embedding(
            input_size = self.encoding.encoding_dim,
            output_size = d_model,
            device = device
        )

        self.transformer = nn.Transformer(
            d_model = d_model,
            nhead = nhead,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers, 
            dim_feedforward = dim_feedforward,
            batch_first=True,
            device = device
        )

        self.linear = nn.Linear(d_model, 1, device=device)

    def forward(self, context: Tensor, queries: Tensor) -> Tensor:
        """
        Args:
            context: Input Tensor with shape [batch_size, num time-series, context length, datapoint dim]
            queries: Input Tensor with shape [batch_size, num time-series, prediction length, datapoint dim]
        Returns:
            output Tensor of shape [batch_size, num time-series, prediction length]
        """

        # Store original shape of queries
        query_shape = queries.shape

        # Embed encoder inputs
        context = self.encoding(context) 
        context = self.embedding(context)

        # Embed decoder inputs
        queries = self.encoding(queries)
        queries = self.embedding(queries)

        # Flatten context and query embeddings
        context = torch.flatten(context, start_dim=1, end_dim=2)
        queries = torch.flatten(queries, start_dim=1, end_dim=2)

        # Pass encoder and decoder inputs to Transformer
        x = self.transformer(
            src = context,
            tgt = queries,
        )
        
        # Pass Transformer outputs through linear layer
        x = self.linear(x)

        # Reshape output to correct shape
        x = x.reshape(query_shape[:-1])

        return x

    # Function to train TimeFusion network
    def train(self,
            data: DataFrame, 
            epochs: int,
            batch_size: int = 64,
            num_batches_per_epoch: int = 50,
            optimizer: optim.Optimizer = None,
            loss_function: Callable = nn.MSELoss()
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
            optimizer = optim.Adam(params=self.parameters(),lr=1e-3, weight_decay=1e-6)
        
        # Batch generator
        data_loader = batch_loader(
            data = data, 
            batch_size = batch_size, 
            num_batches_per_epoch = num_batches_per_epoch, 
            context_length = self.context_length,
            prediction_length = self.prediction_length
        )

        for epoch in range(1, epochs + 1):

            running_loss = 0
            for i, batch in enumerate(data_loader, start = 1):
                # Split data into context, queries and prediction targets
                context, queries, targets = batch

                # Zero gradients
                optimizer.zero_grad()

                # Forward, loss calculation, backward, optimizer step
                predictions = self.forward(context,queries)
                loss = loss_function(predictions,targets)
                loss.backward()
                optimizer.step()

                # Print training statistics
                average_loss = running_loss / epoch
                stat_string = "|" + "="*(30*epoch // epochs) + " "*(30 - (30*epoch // epochs)) + f"|  Batch: {i} / {num_batches_per_epoch}, Epoch: {epoch} / {epochs}, Average Loss: {average_loss:.3f}"
                print("\u007F"*512,"\r",stat_string, end="\r")

            # New line for printing statistics
            print()


# Sine/cos wave encodings of indices and timestamps
class PositionalEncoding(nn.Module):

    def __init__(
            self,
            datapoint_dim: int,
            indices: List[int], 
            timestamps: List[int],
            device: torch.device,
            num_sines: int = 64,
            time_per: List[float] = [
                timedelta(milliseconds=1).total_seconds()*1000,
                timedelta(seconds=1).total_seconds()*1000,
                timedelta(minutes=1).total_seconds()*1000,
                timedelta(days=1).total_seconds()*1000,
                timedelta(weeks=1).total_seconds()*1000,
                timedelta(days=30).total_seconds()*1000,
                timedelta(days=365).total_seconds()*1000,
                timedelta(days=3650).total_seconds()*1000,
            ]
        ) -> None:
        """
        Args:
            datapoint_dim: The number of elements in a single datapoint
            indices: Elements of datapoints to be encoded with sine/cos waves of arbitrary frequency
            timestamps: Elements of datapoints containing timestamps to be encoded with sine/cos waves.
            device: The device for which returned Tensor should be created (e.g. "cpu" or "cuda0")
            num_sines: Total number of sine and cosine waves used to encode indices
            time_per: Period (in millis) of sine/cos waves used to encode timestamps
        """
    
        # Assertions to ensure correct usage
        assert num_sines % 2 == 0, "'num_sines' must be an even number"
        assert all(0 <= idx < datapoint_dim for idx in indices), "'indices' are out of range"
        assert all(0 <= idx < datapoint_dim for idx in timestamps), "'timestamps' are out of range"

        # Init nn.Module base class
        super().__init__()

        # Set instance variables
        self.datapoint_dim = datapoint_dim
        self.device = device
        self.indices = indices
        self.timestamps = timestamps
        self.num_sines = num_sines
        self.time_per = time_per


    def forward(self, source: Tensor) -> Tensor:
        """
        Args:
            source: Input Tensor with shape [batch_size, num time-series, num datapoints, datapoint dim]

        Returns:
            output Tensor of shape [batch_size, num time-series, num datapoints, encoding dim]
        """

        # Shape of output
        output_shape = source.shape[:-1] + tuple([self.encoding_dim])

        # Initialize empty tensor where output values will be stored
        out = torch.zeros(output_shape, dtype=torch.float, device=self.device)

        # Datapoint indexes not to be changed
        unchanged_idx = set(range(source.shape[-1])) - set(self.indices) - set(self.timestamps)

        # Forward all values not to be changed from source to out
        out[:,:,:,:len(unchanged_idx)] = source[:,:,:,list(unchanged_idx)]

        # Encode all indices with cos/sine waves (identical implementation to vanilla transformer sine encodings)
        offset = len(unchanged_idx)
        for i in range(self.num_sines // 2):
            out[:,:,:,offset:offset + len(self.indices)] = torch.sin(source[:,:,:,self.indices]/(10000**(2*i/self.num_sines)))
            offset += len(self.indices)
            out[:,:,:,offset:offset + len(self.indices)] = torch.cos(source[:,:,:,self.indices]/(10000**(2*i/self.num_sines)))
            offset += len(self.indices)

        # Encode all timestamps with cos/sine waves
        for per in self.time_per:
            out[:,:,:,offset:offset + len(self.timestamps)] = torch.sin(source[:,:,:,self.timestamps]*(2*pi/per))
            offset += len(self.timestamps)
            out[:,:,:,offset:offset + len(self.timestamps)] = torch.cos(source[:,:,:,self.timestamps]*(2*pi/per))
            offset += len(self.timestamps)

        return out

    @property
    def encoding_dim(self) -> int:
        """
        Returns: The encoding length of datapoints
        """
        return self.datapoint_dim + len(self.timestamps)*(2*len(self.time_per) - 1) + len(self.indices)*(2*self.num_sines - 1)


# Embed encodings of datapoints to dimension used by Transformer
class Embedding(nn.Module):

    def __init__(
            self,
            input_size: int,
            output_size: int,
            device: torch.device,
            hidden_size: int = 64,
        ):
        """
        Args:
            input_size: Last dimension of input Tensor
            output_size: Last dimension of output Tensor 
            device: The device with which data should be processed (e.g. "cpu" or "cuda0")
            hidden_size: Size of hidden layer in neural network
        """
        
        # Init nn.Module base class
        super().__init__()

        # Set instance variables
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size, device=device)
        self.fc2 = nn.Linear(hidden_size, output_size, device=device)


    def forward(self, source: Tensor) -> Tensor:
        """
        Args:
            source: Input Tensor with shape [batch_size, num time-series, num datapoints, encoding dim]

        Returns:
            output Tensor of shape [batch_size, num time-series, num datapoints, embedding dim]
        """

        x = nn.functional.relu(self.fc1(source))
        x = self.fc2(x)

        return x

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

# NOTE:
# 1. Definitiely should use @torch.no_grad() when measuring inference as this avoids the tracking of gradients, making it 6! times faster