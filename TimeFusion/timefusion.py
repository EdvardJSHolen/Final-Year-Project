# Library imports
import torch

# Module imports
from torch import nn, Tensor
from typing import List
from datetime import timedelta
from math import pi

# Class accessible to end-user
class TimeFusion(nn.Module):

    def __init__(
            self,
            device: torch.device = "cpu",
            ):
        
        # Init nn.Module base class
        super().__init__()

        # Set instance variables
        self.device = device

    def forward():
        return
        #(N, S, E) if batch_first=True
        # NOTE! Must use batch_first=True for Transformer to work with how PositionalEncoding is implemented


# Sine/cos wave encodings of indices and timestamps
class PositionalEncoding(nn.Module):

    def __init__(
            self,
            datapoint_dim: int, 
            indices: List[int], 
            timestamps: List[int],
            num_sines: int = 128,
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
        self.indices = indices
        self.timestamps = timestamps
        self.num_sines = num_sines
        self.time_per = time_per


    def forward(self, source: Tensor, device: torch.device) -> Tensor:
        """
        Args:
            source: Input Tensor with shape [batch_size, num time-series, num datapoints, datapoint dim]

        Returns:
            output Tensor of shape [batch_size, num time-series, num datapoints, encoding dim]
        """

        # Shape of output
        output_shape = source.shape[:-1] + tuple([self.outdim])

        # Initialize empty tensor where output values will be stored
        out = torch.zeros(output_shape, dtype=torch.float64, device=device)

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
    def outdim(self) -> int:
        """
        Returns: The encoding length of datapoints
        """
        return self.datapoint_dim + len(self.timestamps)*(2*len(self.time_per) - 1) + len(self.indices)*(2*self.num_sines - 1)


# Embedding
class Embedding(nn.Module):

    def __init__(
            self,
            device: torch.device = "cpu",
            ):
        
        # Init nn.Module base class
        super().__init__()

        # Set instance variables
        self.device = device

    def forward():
        return
        #(N, S, E) if batch_first=True
        # NOTE! Must use batch_first=True for Transformer to work with how PositionalEncoding is implemented


# TODO:
# 1. Investigate where it is beneficial to add @torch.no_grad() decorator

# NOTE:
# 1. Definitiely should use @torch.no_grad() when measuring inference as this avoids the tracking of gradients, making it faster.