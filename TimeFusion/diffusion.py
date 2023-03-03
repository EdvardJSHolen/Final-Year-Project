# Library imports
import torch
import random
import math
import pandas as pd
import numpy as np

# Module imports
from torch import Tensor
from typing import Generator, Iterable

# Functions for generating training batches and sampling starting points
def batch_loader(
        device: torch.device,
        data: pd.DataFrame, 
        batch_size: int, 
        num_batches_per_epoch: int, 
        context_length: int,
        prediction_length: int,
        diff_steps: int,
        betas: Iterable[float] = None
    ) -> Generator[Tensor,Tensor,Tensor]:
    """
    Args:
        data: Pandas DataFrame with timestamps as the index and a column for each time-series. Cells should be filled with
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

    if betas is None:
        betas = np.linspace(1e-4, 0.1, diff_steps)

    alphas = 1.0 - betas
    bar_alphas = np.cumprod(alphas)
    
    
    # Find minimum and maximum index at which division between context and target can be made
    min_idx = 0
    max_idx = len(data) - 1
    for column in data:
        min_idx = max(min_idx,data[column].dropna().index[:context_length + 1][-1])
        max_idx = min(max_idx,data[column].dropna().index[-prediction_length:][0])
    
    assert min_idx <= max_idx , "Not enough data provided for given context and prediction length"

    # Shuffle indices to get random order for each batch and epoch
    shuffled_indices = list(data.index[pd.Series(data.index).between(min_idx,max_idx)])
    random.shuffle(shuffled_indices)


    for k in range(num_batches_per_epoch):

        if batch_size * k >= len(shuffled_indices):
            print(f"{num_batches_per_epoch} was expected, but only {math.ceil(len(shuffled_indices)/batch_size)} can be create with given data!")
            break

        # Create Tensors to hold batch
        context_tensor = torch.empty(0,device=device)
        query_tensor = torch.empty(0,device=device)
        target_tensor = torch.empty(0,device=device)

        indices = shuffled_indices[batch_size * k:batch_size * (k+1)]

        for idx in indices:

            # Split data at index
            before = data[data.index < idx]
            after = data[data.index >= idx]

            # Create context vector
            context = torch.empty(0,device=device)
            # Add values, timestamps, time-series index, datapoint index and diffusion index
            for j, column in enumerate(data.columns):
                col_values = before[column].dropna()
                col_tensor = torch.tensor([[col_values.iloc[i - context_length],col_values.index[i - context_length],i,j,0] for i in range(0, context_length)],device=device,dtype=torch.float32)
                context = torch.cat((context,col_tensor[None,:]))

            # Select random diffusion step n
            n = random.randrange(1, diff_steps + 1)

            # Non-diffused queries
            query = torch.empty(0,device=device)
            # Add values, timestamps, time-series index, datapoint index and diffusion index
            for j, column in enumerate(data.columns):
                col_values = after[column].dropna()
                col_tensor = torch.tensor([[col_values.iloc[i],col_values.index[i],i + context_length,j,n] for i in range(0, prediction_length)],device=device,dtype=torch.float32)
                query = torch.cat((query,col_tensor[None,:]))

            # Sample targets from N(0,I)
            #target = torch.empty((len(data.columns),prediction_length),device=device).normal_()
            target = query[:,:,0]

            # Diffuse queries
            #query[:,:,0] = math.sqrt(bar_alphas[n - 1])*query[:,:,0] + math.sqrt(1-bar_alphas[n - 1])*target

            # Concatenate data with rest of batch
            context_tensor = torch.cat((context_tensor,context[None,:]))
            query_tensor = torch.cat((query_tensor,query[None,:]))
            target_tensor = torch.cat((target_tensor,target[None,:]))

        # Yield a single batch
        yield (context_tensor, query_tensor, target_tensor)


# TODO:
# 1. Need to make this data generator into a class to avoid pre-processing of data at the beginning of every epoch.
# 2. Do I need to set device on which these Tensors are handled. YES! DO IT NOW!
# 3. REALLY need to optimize this function