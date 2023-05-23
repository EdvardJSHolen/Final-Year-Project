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
        end_padding: int = 0,
        ts_columns: List[int] = None,
    ):
        # Make a copy of the input dataframe
        self.data = data.copy()
        self.data.index = pd.to_datetime(self.data.index)
        self.data.columns = list(range(self.data.shape[1]))
        
        # Save instance variables
        self.context_length = context_length
        self.end_padding = end_padding
        self.ts_columns = ts_columns
        if self.ts_columns is None:
            self.ts_columns = list(range(self.data.shape[1]))

        # Store the time series in a Tensor
        self.tensor_data = torch.tensor(self.data.to_numpy(),dtype=torch.float32)

    def add_timestamp_encodings(self,
        timestamp_encodings: List[Callable] = [
            lambda x: math.sin(2*math.pi*x.hour / 24),
            lambda x: math.sin(2*math.pi*x.weekday() / 7),
            lambda x: math.sin(2*math.pi*x.day / calendar.monthrange(x.year, x.month)[1]),
            lambda x: math.sin(2*math.pi*x.month / 12),
            lambda x: math.cos(2*math.pi*x.hour / 24),
            lambda x: math.cos(2*math.pi*x.weekday() / 7),
            lambda x: math.cos(2*math.pi*x.day / calendar.monthrange(x.year, x.month)[1]),
            lambda x: math.cos(2*math.pi*x.month / 12),
        ]
    ) -> None:
        
        # Add columns of encodings
        for encoding in timestamp_encodings:
            self.data.insert(self.data.shape[1], self.data.shape[1], [encoding(x) for x in self.data.index])

        # Update data tensor
        self.tensor_data = torch.tensor(self.data.to_numpy(),dtype=torch.float32)

    def get_sample_tensor(self, idx: int) -> Tensor:
        context = torch.clone(self.tensor_data[idx : idx + self.context_length].T)
        return context

    def __len__(self) -> int:
        return max(0, self.tensor_data.shape[0] - self.context_length - 1 - self.end_padding)

    def __getitem__(self, idx: int) -> Tensor:
        context = torch.clone(self.tensor_data[idx : idx + self.context_length].T)
        target = torch.clone(self.tensor_data[idx + self.context_length, self.ts_columns])
        return context, target