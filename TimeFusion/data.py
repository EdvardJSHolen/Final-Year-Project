import torch
import math
import calendar
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import List, Callable
from pandas import DatetimeIndex
from torch import Tensor

class TimeFusionDataset(Dataset):
    def __init__(self,
        data: pd.DataFrame, 
        context_length: int,
        prediction_length: int,
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
    ) -> None:
        
        # Save instance variables
        self.context_length = context_length
        self.prediction_length = prediction_length

        # Make a copy of the input dataframe
        self.data_copy = data.copy()
        self.data_copy.index = pd.to_datetime(self.data_copy.index)

        # Add columns encoding absolute time-position
        for encoding in timestamp_encodings:
            self.data_copy.insert(len(self.data_copy.columns), len(self.data_copy.columns), [encoding(x) for x in self.data_copy.index], allow_duplicates=True)

        # Add column for diffusion index
        #self.data_copy[len(self.data_copy.columns)] = 0
        self.data_copy[len(self.data_copy.columns)] = -1

        # Positions at which to cut each tensor when creating token
        indices = np.cumsum(data.notnull().values, axis=0)
        valid_indices = indices[(np.min(indices,axis=1) > self.context_length) & (np.max(indices - np.max(indices,axis=0),axis=1) <= -self.prediction_length)]
        self.indices = valid_indices

        # Split each individual time-series into a separate DataFrame
        time_series = [self.data_copy.iloc[:,[i] + list(range(data.shape[1],self.data_copy.shape[1]))].dropna() for i in range(len(data.columns))]
        # Store the time series in a Tensor
        self.time_series = torch.empty(
            size = (len(time_series),np.max(indices),time_series[0].shape[1]),
            dtype = torch.float32,
        )
        for i, ts in enumerate(time_series):
            self.time_series[i,:ts.shape[0]] = torch.tensor(ts.values)

    def __len__(self) -> int:
        return self.indices.shape[0]

    def __getitem__(self, idx:int) -> Tensor:
        col_segments = []
        for col in range(self.time_series.shape[0]):
            midpoint = self.indices[idx,col]
            col_segments.append(self.time_series[col, midpoint - self.context_length:midpoint + self.prediction_length])
        return torch.stack(col_segments)
    

    def get_sample_tensor(self, 
        sample_indices: List[DatetimeIndex], 
        timestamp_encodings: List[Callable]
    ) -> Tensor:

        # Ensure that enough indices have been passed to the function
        assert len(sample_indices) == self.time_series.shape[0]
        assert len(sample_indices[0]) == self.prediction_length

        # Determine what data should be used as context
        context_df = self.data_copy[self.data_copy.index < np.min(sample_indices)]

        # Convert to context data to Tensor
        context = []
        for col in range(self.time_series.shape[0]):
            col_values = context_df.iloc[:,[col] + list(range(self.time_series.shape[0],context_df.shape[1]))].dropna().values
            context.append(col_values[-self.context_length:])
        context = torch.tensor(
            data = np.array(context),
            dtype = torch.float32,
        )

        # Calculate the subquery for each time series
        subqueries = []
        for indices in sample_indices:
            # Initialise dataframe for query data
            queries_df = pd.DataFrame(0, index=indices, columns = [0])

            # Time encodings
            for encoding in timestamp_encodings:
                queries_df.insert(queries_df.shape[1], queries_df.shape[1], [encoding(x) for x in indices])

            # Diffusion index
            queries_df[queries_df.shape[1]] = 0

            subqueries.append(
                torch.tensor(
                    data = queries_df.values,
                    dtype = torch.float32
                )
            )

        # Combine subqueries to single Tensor
        queries = torch.stack(subqueries)
        
        # Return token consisting of context and queries
        return torch.cat((context, queries), dim = 1)
