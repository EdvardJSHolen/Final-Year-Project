# Library imports
import torch
import random
import math
import pandas as pd
import numpy as np

# Module imports
from torch import Tensor
from typing import List

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

        if betas is None:
            self.betas = np.linspace(1e-4, 0.1, diff_steps)

        self.alphas = 1.0 - self.betas
        self.bar_alphas = np.cumprod(self.alphas)

        # Ensure data is sorted
        data.sort_index(inplace=True)

        # Find minimum and maximum index at which division between context and target can be made
        min_idx = data.index[0]
        max_idx = data.index[-1]
        for column in data:
            min_idx = max(min_idx, data[column].dropna().index[:context_length + 1][-1])
            max_idx = min(max_idx, data[column].dropna().index[-prediction_length:][0])
    
        assert min_idx <= max_idx , f"Not enough data provided for given context and prediction length"

        # Get all integer indices at which it is possible to split data
        indices = list(pd.Series(range(len(data)))[pd.Series(data.index).between(min_idx,max_idx)])

        # Calculate context, queries and targets for each valid index
        self.contexts = torch.empty(0, device = device)
        self.queries = torch.empty(0, device = device)
        self.targets = torch.empty(0, device = device)
        for idx in indices:

            # Split data at index
            context_data = data.iloc[:idx]
            query_data = data.iloc[idx:]

            # Create context Tensor
            context = torch.empty(0, device = device)
            for column in context_data.columns:
                col_values = context_data[column].dropna()
                col_tensor = torch.tensor(
                    [
                        list(col_values.iloc[-context_length:]), # Value
                        list(np.array(col_values.index[-context_length:] - data.index[idx])/context_length), # Timestamp
                        list(np.zeros(shape=(context_length))) # Diffusion index
                    ],
                    dtype = torch.float32,
                    device = device
                ).t()
                context = torch.cat((context, col_tensor.unsqueeze(0)))
            self.contexts = torch.cat((self.contexts, context.unsqueeze(0)))

            # Create query Tensor (non-diffused)
            query = torch.empty(0, device = device)
            for column in query_data.columns:
                col_values = query_data[column].dropna()
                col_tensor = torch.tensor(
                    [
                        list(col_values.iloc[:prediction_length]), # Value
                        list(np.array(col_values.index[:prediction_length] - data.index[idx])/context_length), # Timestamp
                        list(np.zeros(shape=(prediction_length))) # Diffusion index
                    ],
                    dtype = torch.float32,
                    device = device
                ).t()
                query = torch.cat((query, col_tensor.unsqueeze(0)))
            self.queries = torch.cat((self.queries, query.unsqueeze(0)))

            # Combine queries and context
            self.tokens = torch.cat((self.contexts,self.queries),dim = -2)

            # Create target Tensor
            target = torch.clone(query[:,:,0])
            self.targets = torch.cat((self.targets, target.unsqueeze(0)))
    

    def __iter__(self):
        return self._generator()
    

    def _generator(self):

        # Get random ordering of samples
        shuffled_indices = list(range(len(self.tokens)))
        random.shuffle(shuffled_indices)


        # Yield batches until all data has been looped through
        i = 0
        while i * self.batch_size < len(shuffled_indices):

            batch_indices = shuffled_indices[i * self.batch_size : (i+1) * self.batch_size]

            # Fetch data from list of samples
            tokens = self.tokens[batch_indices]

            # Sample targets from N(0,I)
            target_tensor = torch.empty(size=tokens[:,:,self.context_length:,0].shape,device=self.device).normal_()
            #target_tensor = self.targets[batch_indices]

            # Diffuse data
            for k in range(tokens.shape[0]):
                n = random.randrange(1, self.diff_steps + 1)
                tokens[k,:,self.context_length:,0] = math.sqrt(self.bar_alphas[n - 1])*tokens[k,:,self.context_length:,0] + math.sqrt(1-self.bar_alphas[n - 1])*target_tensor[k]
                tokens[k,:,self.context_length:,-1] = 2*n / self.diff_steps - 1

            # Yield a single batch
            yield (tokens, target_tensor)

            # Increment counter
            i += 1


    @property
    def num_batches(self) -> int:
        """
        Returns: The number of batches per epoch
        """
        return math.ceil(len(self.tokens) / self.batch_size)

