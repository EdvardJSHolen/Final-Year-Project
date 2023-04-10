# Library imports
import torch
import random
import math
import calendar
import pandas as pd
import numpy as np

# Module imports
from torch import Tensor
from typing import List, Callable

# Iterable class which batches data
class BatchLoader():
        
    def __init__(
        self,
        data: pd.DataFrame, 
        context_length: int,
        prediction_length: int,
        batch_size: int = 64, 
        diff_steps: int = 100,
        betas: List[float] = None,
        device: torch.device = torch.device("cpu"),
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
        lazy_init: bool = False
    ) -> None:
        
        """
        Args:
            data: Pandas DataFrame with timestamps as the index and a column for each time-series.
            context_length: The number of datapoints for each time-series that should be given as context to predictor
            prediction_length: How many steps into the future to predict
            batch_size: The number of samples to process at the same time.
            diff_steps: The number of diffusion steps
            betas: A schedule containing the beta values for n = {1,...,diff_steps}
            device: The device on which computations should be performed (e.g. "cpu" or "cuda0").
        Returns:
            Generator giving Tensors for inputs and targets of shape [batch_size, total length, num time-series, datapoint dim], 
            and [batch_size, prediction length, num time-series] respectively.
        """

        # Set instance variables
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.diff_steps = diff_steps
        self.betas = betas
        self.device = device
        self.lazy_init = lazy_init

        if betas is None:
            self.betas = np.linspace(1e-4, 0.1, diff_steps)

        self.alphas = 1.0 - self.betas
        self.bar_alphas = np.cumprod(self.alphas)

        # Make a copy of the input dataframe
        data = data.copy()
        data.index = pd.to_datetime(data.index)

        # Ensure data is sorted
        data.sort_index(inplace=True)

        # Rename columns
        data.columns = list(range(len(data.columns)))
        columns = list(data.columns)

        # Add columns encoding absolute time-position
        for encoding in timestamp_encodings:
            data.insert(len(data.columns), len(data.columns), [encoding(x) for x in data.index])

        # Add column for diffusion index
        data[len(data.columns)] = 0

        # Split each individual time-series into a separate DataFrame
        time_series = [data[[column] + list(data.columns[len(columns):])].dropna() for column in columns]


        # Find minimum and maximum index at which division between context and target can be made
        min_idx = data.index[0]
        max_idx = data.index[-1]
        for ts in time_series:
            min_idx = max(min_idx, ts.index[context_length + 1])
            max_idx = min(max_idx, ts.index[-prediction_length])
    
        assert min_idx <= max_idx , f"Not enough data provided for given context and prediction length"

        if lazy_init:
            self.time_series = time_series
            self.indices = list(data.index[(data.index >= min_idx) & (data.index <= max_idx)])
        else:
            # Calculate input tokens
            contexts = []
            queries = []
            for idx in data.index[(data.index >= min_idx) & (data.index <= max_idx)]:
                # Fetch historical and future datapoints for timestamp index
                contexts.append([ts[:idx][-self.context_length - 1: -1].values for ts in time_series])
                queries.append([ts[idx:][:self.prediction_length].values for ts in time_series])

            contexts = torch.tensor(
                np.array(contexts),
                dtype = torch.float32,
                device = device
            )

            queries = torch.tensor(
                np.array(queries),
                dtype = torch.float32,
                device = device
            )

            # Save tokens as torch.Tensor
            self.tokens = torch.cat((contexts,queries), dim = 2)


    def __iter__(self):
        return self._generator()
    

    def _generator(self):

        if self.lazy_init:
            # Get random ordering of samples
            shuffled_indices = self.indices.copy()
            random.shuffle(shuffled_indices)

            # Yield batches until all data has been looped through
            i = 0
            while i * self.batch_size < len(shuffled_indices):

                batch_indices = shuffled_indices[i * self.batch_size : (i+1) * self.batch_size]

                # Calculate input tokens
                contexts = []
                queries = []
                for idx in batch_indices:
                    # Fetch historical and future datapoints for timestamp index
                    contexts.append([ts[:idx][-self.context_length - 1: -1].values for ts in self.time_series])
                    queries.append([ts[idx:][:self.prediction_length].values for ts in self.time_series])

                contexts = torch.tensor(
                    np.array(contexts),
                    dtype = torch.float32,
                    device = self.device
                )

                queries = torch.tensor(
                    np.array(queries),
                    dtype = torch.float32,
                    device = self.device
                )

                # Sample targets from N(0,I)
                targets = torch.empty(size=queries[:,:,:,0].shape,device=self.device).normal_()

                # Diffuse data
                for k in range(queries.shape[0]):
                    n = random.randrange(1, self.diff_steps + 1)
                    queries[k,:,:,0] = math.sqrt(self.bar_alphas[n - 1])*queries[k,:,:,0] + math.sqrt(1-self.bar_alphas[n - 1])*targets[k]
                    queries[k,:,:,-1] = 2*n / self.diff_steps - 1

                # Combine contexts and queries into input tokens
                tokens = torch.cat((contexts,queries), dim = 2)

                # Yield a single batch
                yield (tokens, targets)

                # Increment counter
                i += 1
            
        else:
            # Get random ordering of samples
            shuffled_indices = list(range(self.tokens.shape[0]))
            random.shuffle(shuffled_indices)

            # Yield batches until all data has been looped through
            i = 0
            while i * self.batch_size < len(shuffled_indices):

                batch_indices = shuffled_indices[i * self.batch_size : (i+1) * self.batch_size]

                # Fetch data from list of samples
                tokens = self.tokens[batch_indices]

                # Sample targets from N(0,I)
                targets = torch.empty(size=tokens[:,:,self.context_length:,0].shape,device=self.device).normal_()

                # Diffuse data
                for k in range(tokens.shape[0]):
                    n = random.randrange(1, self.diff_steps + 1)
                    tokens[k,:,self.context_length:,0] = math.sqrt(self.bar_alphas[n - 1])*tokens[k,:,self.context_length:,0] + math.sqrt(1-self.bar_alphas[n - 1])*targets[k]
                    tokens[k,:,self.context_length:,-1] = 2*n / self.diff_steps - 1

                # Yield a single batch
                yield (tokens, targets)

                # Increment counter
                i += 1


    @property
    def num_batches(self) -> int:
        """
        Returns: The number of batches per epoch
        """
        if self.lazy_init:
            return math.ceil(len(self.indices) / self.batch_size)
        else:
            return math.ceil(len(self.tokens) / self.batch_size)

