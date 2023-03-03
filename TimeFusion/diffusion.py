# Library imports
import torch
import random
import math
import pandas as pd

# Module imports
from torch import Tensor
from typing import Generator

# Functions for generating training batches and sampling starting points
def batch_loader(
        data: pd.DataFrame, 
        batch_size: int, 
        num_batches_per_epoch: int, 
        context_length: int,
        prediction_length: int,
        diff_steps: int,
        # alpha/beta schedule
    ) -> Generator[Tensor,Tensor,Tensor]:
    """
    Args:
        data: Pandas DataFrame with timestamps as the index and a column for each time-series. Cells should be filled with
        num_batches_per_epoch: The number of batches to train on in a given epoch.
        batch_size: The number of samples to process at the same time.
    Returns:
        Generator giving Tensors for context, queries and targets of shape [batch_size,num time-series, num datapoints, datapoint dim], where num 
        datapoints is context length for context and prediction length for queries and targets, and data point dim is replaced with 1 for targets
    """
    
    # Find minimum and maximum index at which division between context and target can be made
    min_idx = 0
    max_idx = len(data) - 1
    for column in data:
        min_idx = max(min_idx,data[column].dropna().index[:context_length][-1])
        max_idx = min(max_idx,data[column].dropna().index[-prediction_length:][0])
    
    assert min_idx < max_idx , "Not enough data provided for given context and prediction length"

    # Shuffle indices to get random order for each batch and epoch
    shuffled_indices = data.index[pd.Series(data.index).between(min_idx,max_idx)]
    random.shuffle(shuffled_indices)


    for k in range(num_batches_per_epoch):

        if batch_size * k >= len(idx):
            print(f"{num_batches_per_epoch} was expected, but only {math.ceil(len(shuffled_indices)/batch_size)} can be create with given data!")
            break

        # Create Tensors to hold batch
        context_tensor = torch.empty(0)
        query_tensor = torch.empty(0)
        target_tensor = torch.empty(0)

        indices = shuffled_indices[batch_size * k:batch_size * (k+1)]

        for idx in indices:

            # Create context vector
            context = torch.empty(0)
            # Add values, timestamps, time-series index, datapoint index and diffusion index
            for j, column in enumerate(data.columns):
                col_values = data[column].dropna()
                col_tensor = Tensor([[col_values.iloc[i - context_length],col_values.index[i - context_length],j,i,0] for i in range(idx, context_length + idx)])
                context = torch.cat((context,col_tensor[None,:]))

            # Select random diffusion step n
            n = random.randrange(1, diff_steps + 1)


            # NEED TO DECIDE WHETHER CONTEXT OR QUERIES ARE GOING TO OWN IDX
            # Non-diffused queries
            clean_queries = torch.empty(0)
            # Add values, timestamps, time-series index, datapoint index and diffusion index
            for j, column in enumerate(data.columns):
                col_values = data[column].dropna()
                col_tensor = Tensor([[col_values.iloc[i - context_length],col_values.index[i - context_length],j,i,n] for i in range(idx, context_length + idx)])
                context = torch.cat((context,col_tensor[None,:]))

            # Concatenate data with rest of batch
            context_tensor = torch.cat((context_tensor,context[None,:]))
            query_tensor = torch.cat((query_tensor,query[None,:]))
            target_tensor = torch.cat((context_tensor,target[None,:]))

        # Yield a single batch
        yield context_tensor, query_tensor, target_tensor


# TODO:
# 1. Need to make this data generator into a class to avoid pre-processing of data at the beginning of every epoch.
# 2. Do I need to set device on which these Tensors are handled?