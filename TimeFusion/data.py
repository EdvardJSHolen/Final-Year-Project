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

        # Save the mask
        self.mask = torch.tensor(data.isna().to_numpy().T)

        # Do interpolation
        self.data_copy = self.data_copy.interpolate(method="linear").fillna(0)

        # Add columns encoding absolute time-position
        for encoding in timestamp_encodings:
            self.data_copy.insert(len(self.data_copy.columns), len(self.data_copy.columns), [encoding(x) for x in self.data_copy.index], allow_duplicates=True)

        # Add column for diffusion index
        self.data_copy[len(self.data_copy.columns)] = 0

        self.indices = list(range(context_length,data.shape[0] - prediction_length))#data.index[context_length:-prediction_length]

        time_series = [self.data_copy.iloc[:,[i] + list(range(data.shape[1],self.data_copy.shape[1]))].to_numpy() for i in range(len(data.columns))]

        self.tensor_data = torch.tensor(
            data = np.stack(time_series,axis = 0),
            dtype = torch.float32,
        )
                                  

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx:int) -> Tensor:
        tokens = self.tensor_data[:,self.indices[idx]-self.context_length:self.indices[idx]+self.prediction_length]
        mask = self.mask[:,self.indices[idx]-self.context_length:self.indices[idx]+self.prediction_length]
        
        return  torch.clone(tokens), torch.clone(mask)
    
