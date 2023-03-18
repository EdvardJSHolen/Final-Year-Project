# Library imports
import torch
import math
import pandas as pd
import numpy as np

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
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.start_length = start_length
        self.device = device
        self.diff_steps = diff_steps
    
        if betas is None:
            self.betas = np.linspace(1e-4, 0.1, diff_steps)

        self.alphas = 1.0 - self.betas
        self.bar_alphas = np.cumprod(self.alphas)

        self.encoding = PositionalEncoding(
            datapoint_dim = datapoint_dim,
            indices = indices,
            timestamps = timestamps,
            device = device,
            num_sines = 64
        )

        self.embedding = Embedding(
            input_size = self.encoding.encoding_dim,
            output_size = int(d_model/2),
            hidden_size = 256,
            device = device,
        )

        self.transformer = nn.Transformer(
            d_model = d_model,
            nhead = nhead,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers, 
            dim_feedforward = dim_feedforward,
            batch_first=True,
            device = device,
            dropout=0
        )

        #self.linear = nn.Linear(d_model, 1, device=device)
        self.linear = nn.Linear(d_model, 2, device=device)
        self.linear0  = nn.Linear(d_model, d_model, device=device)

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
        context = torch.permute(context, (0, 2, 1, 3))
        queries = torch.permute(queries, (0, 2, 1, 3))
        context = torch.flatten(context, start_dim=2, end_dim=3)
        queries = torch.flatten(queries, start_dim=2, end_dim=3)
        #context = torch.flatten(context, start_dim=1, end_dim=2)
        #queries = torch.flatten(queries, start_dim=1, end_dim=2)

        # Pass encoder and decoder inputs to Transformer
        x = self.transformer(
           src = context,
           tgt = queries,
        )
        
        # Pass Transformer outputs through linear layer
        #x = nn.functional.relu(self.linear0(x))
        x = self.linear(x)

        # Reshape output to correct shape
        x = x.reshape(query_shape[:-1])

        x = x[:,:,self.start_length:]

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
            optimizer = optim.Adam(params=self.parameters(),lr=2e-4,weight_decay=1e-5)


        train_loader = BatchLoader(
            data = train_data,
            batch_size = batch_size,
            context_length = self.context_length,
            prediction_length = self.prediction_length,
            start_length = self.start_length,
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
                start_length = self.start_length,
                diff_steps = self.diff_steps,
                device = self.device
            )
        

        for epoch in range(1, epochs + 1):

            running_loss = 0
            for i, batch in enumerate(train_loader, start = 1):
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
                running_loss += loss.item()
                average_loss = running_loss / i
                stat_string = "|" + "="*(30*i // train_loader.num_batches) + " "*(30 - (30*i // train_loader.num_batches)) + f"|  Batch: {i} / {train_loader.num_batches}, Epoch: {epoch} / {epochs}, Average Loss: {average_loss:.4f}"
                print("\u007F"*512,stat_string,end="\r")

            if val_data is not None:
                with torch.no_grad():
                    running_loss = {key:0 for key in val_metrics.keys()}
                    for i, batch in enumerate(val_loader, start = 1):
                        context, queries, targets = batch
                        predictions = self.forward(context,queries)
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
                    list(col_values.index[-self.context_length:] - data.index[-1]), # Timestamp
                    list(range(self.context_length)), # Datapoint index
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
                    list(np.array(range(1,self.prediction_length + 1))), # Timestamp
                    list(range(self.context_length,self.prediction_length + self.context_length)), # Datapoint index
                    #list(np.full(shape=(prediction_length + start_length),fill_value=j)), # Time-series index
                    list(np.zeros(shape=(self.prediction_length))) # Diffusion index
                ],
                dtype = torch.float32,
                device = self.device
            ).t()
            query = torch.cat((query, col_tensor[None,:]))
        query = torch.cat((context[:,-self.start_length:,:],query),dim=1) # Add start token


        # Copy context and query tensor to be of same size as batch size (HACK)
        context = context.unsqueeze(0).repeat([batch_size] + [1]*context.dim())
        query = query.unsqueeze(0).repeat([batch_size] + [1]*query.dim())

        # Sample
        samples = torch.empty(0, device = self.device)
        iterations = torch.empty(0, device = self.device)
        iterations2 = torch.empty(0, device = self.device)
        while samples.shape[0] < num_samples:

            # Make sure we do not make too many predictions if num_samples % batch_size is not equal to 0
            if num_samples - samples.shape[0] < batch_size:
                query = query[:num_samples - samples.shape[0]]
                context = context[:num_samples - samples.shape[0]]

            # Sample initial white noise
            query[:,:,self.start_length:,0] = torch.empty(size=query[:,:,self.start_length:,0].shape,device=self.device).normal_()

            # Compute each diffusion step
            for n in range(self.diff_steps,0,-1):

                # Set diffusion step
                query[:,:,self.start_length:,-1] = n

                iterations = torch.cat((iterations, query))

                # Calculate output
                output = self.forward(context,query)

                iterations2 = torch.cat((iterations2, output))

                # Give new value to query
                if n > 1:
                    z = torch.empty(size=query[:,:,self.start_length:,0].shape,device=self.device).normal_()
                else: 
                    z = torch.zeros(size=query[:,:,self.start_length:,0].shape,device=self.device)

                query[:,:,self.start_length:,0] = (1/math.sqrt(self.alphas[n-1]))*(query[:,:,self.start_length:,0] - (self.betas[n-1]/math.sqrt(1-self.bar_alphas[n-1]))*output) + math.sqrt(self.betas[n-1])*z

            # Add denoised queries to samples
            #samples = torch.cat((samples, query[:,:,self.start_length:]))
            samples = torch.cat((samples, query))

        return samples, iterations, iterations2

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
                timedelta(minutes=1).total_seconds()*1000,
                timedelta(days=1).total_seconds()*1000,
                timedelta(weeks=1).total_seconds()*1000,
                timedelta(days=30).total_seconds()*1000,
                timedelta(days=365).total_seconds()*1000,
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
# 10. Add time to training statistics
# 11. Need to consolidate beta schedules, currently define them two places.
# 12. NEED TO SET model.eval() ????


# NOTE:
# 1. Definitiely should use @torch.no_grad() when measuring inference as this avoids the tracking of gradients, making it 6! times faster