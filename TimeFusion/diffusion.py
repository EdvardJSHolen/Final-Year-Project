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
        start_length: int = 0,
        batch_size: int = 64, 
        diff_steps: int = 100,
        betas: List[float] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        
        """
        Args:
            data: Pandas DataFrame with timestamps as the index and a column for each time-series. Cells should be filled with.
            num_batches_per_epoch: The number of batches to train on in a given epoch.
            batch_size: The number of samples to process at the same time.
            context_length: The number of datapoints for each time-series that should be given as context to predictor
            prediction_length: How many steps into the future to predict
            diff_steps: The number of diffusion steps
            betas: A schedule containing the beta values for n = {1,...,diff_steps}
        Returns:
            Generator giving Tensors for context, queries and targets of shape [batch_size,num time-series, context length, datapoint dim], 
            [batch_size,num time-series, prediction length, datapoint dim] and [batch_size,num time-series, prediction length] respectively.
        """

        self.batch_size = batch_size
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.start_length = start_length
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
            min_idx = max(min_idx, data[column].dropna().index[:context_length + 1][-1], data[column].dropna().index[:start_length + 1][-1])
            max_idx = min(max_idx, data[column].dropna().index[-prediction_length:][0])
    
        assert min_idx <= max_idx , f"Not enough data provided for given context and prediction length"

        # Get all indices at which it is possible to split data
        #indices = list(data.index[pd.Series(data.index).between(min_idx,max_idx)])
        # Get all integer indices at which it is possible to split data
        indices = list(pd.Series(range(len(data)))[pd.Series(data.index).between(min_idx,max_idx)])

        # Calculate context, queries and targets for each valid index
        self.contexts = torch.empty(0, device = device)
        self.queries = torch.empty(0, device = device)
        self.targets = torch.empty(0, device = device)
        for idx in indices:

            # Split data at index
            #context_data = data[data.index < idx]
            #query_data = data[data.index >= idx - start_length] # NEED TO FIX THIS TO WORK WITH IRREGULAR DATA, can be done using iloc?
            context_data = data.iloc[:idx]
            query_data = data.iloc[idx:]

            # Create context Tensor
            context = torch.empty(0, device = device)
            for j, column in enumerate(context_data.columns):
                col_values = context_data[column].dropna()
                col_tensor = torch.tensor(
                    [
                        list(col_values.iloc[-context_length:]), # Value
                        list(col_values.index[-context_length:] - data.index[idx]), # Timestamp
                        list(range(context_length)), # Datapoint index
                        #list(np.full(shape=(context_length),fill_value=j)), # Time-series index
                        list(np.zeros(shape=(context_length))) # Diffusion index
                    ],
                    dtype = torch.float32,
                    device = device
                ).t()
                context = torch.cat((context, col_tensor[None,:]))
            self.contexts = torch.cat((self.contexts, context[None,:]))

            # Create query Tensor (non-diffused)
            query = torch.empty(0, device = device)
            for j, column in enumerate(query_data.columns):
                col_values = query_data[column].dropna()
                col_tensor = torch.tensor(
                    [
                        list(col_values.iloc[:prediction_length]), # Value
                        list(col_values.index[:prediction_length] - data.index[idx]), # Timestamp
                        list(range(context_length, context_length + prediction_length)), # Datapoint index
                        #list(np.full(shape=(prediction_length + start_length),fill_value=j)), # Time-series index
                        list(np.zeros(shape=(prediction_length))) # Diffusion index
                    ],
                    dtype = torch.float32,
                    device = device
                ).t()
                query = torch.cat((query, col_tensor[None,:]))
            query = torch.cat((context[:,-start_length:,:],query),dim=1) # Add start token
            self.queries = torch.cat((self.queries, query[None,:]))

            # Create target Tensor
            target = torch.clone(query[:,start_length:,0])
            self.targets = torch.cat((self.targets, target[None,:]))
    

    def __iter__(self):
        return self._generator()
    

    def _generator(self):

        # Get random ordering of samples
        shuffled_indices = list(range(len(self.contexts)))
        random.shuffle(shuffled_indices)


        # Yield batches until all data has been looped through
        i = 0
        while i * self.batch_size < len(shuffled_indices):

            batch_indices = shuffled_indices[i * self.batch_size : (i+1) * self.batch_size]

            # Fetch data from list of samples
            context_tensor = self.contexts[batch_indices]
            query_tensor = self.queries[batch_indices]

            # Sample targets from N(0,I)
            target_tensor = torch.empty(size=query_tensor[:,:,self.start_length:,0].shape,device=self.device).normal_()
            #target_tensor = self.targets[batch_indices]

            # Diffuse data
            for k in range(query_tensor.shape[0]):
                n = random.randrange(1, self.diff_steps + 1)
                query_tensor[k,:,self.start_length:,0] = math.sqrt(self.bar_alphas[n - 1])*query_tensor[k,:,self.start_length:,0] + math.sqrt(1-self.bar_alphas[n - 1])*target_tensor[k]
                query_tensor[k,:,self.start_length:,-1] = n
            #query_tensor[:,:,self.start_length:,0] = torch.zeros(query_tensor[:,:,self.start_length:,0].shape)

            # Yield a single batch
            yield (context_tensor, query_tensor, target_tensor)

            # Increment counter
            i += 1


    @property
    def num_batches(self) -> int:
        """
        Returns: The number of batches per epoch
        """
        return math.ceil(len(self.contexts) / self.batch_size)


# TODO:
# 1. Need to handle situations where start_length is shorter than context_length