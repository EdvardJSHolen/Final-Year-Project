import torch
import math
import calendar
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Callable
from torch import Tensor

class TimeFusionDataset(Dataset):

    def __init__(
        self,
        data: pd.DataFrame, 
        context_length: int,
        pred_columns: List[int] = None,
    ):
        """
        Args:
            data: Pandas dataframe with time-series and covariates as columns
            context_length: Number of past time steps to use as context
            pred_columns: The numerical columns indices to use as prediction targets
        """
        
        # Make a copy of the dataframe
        self.data = data.copy()
        self.data.index = pd.to_datetime(self.data.index)
        self.data.columns = list(range(self.data.shape[1]))
        
        # Save instance variables
        self.context_length = context_length

        self.pred_columns = pred_columns
        if self.pred_columns is None:
            self.pred_columns = self.data.columns
        self.cov_columns = list(set(self.data.columns) - set(self.pred_columns))

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
        
        """
        Args:
            timestamp_encodings: List of encoding functions to apply to the timestamp index
        """
        
        # Add columns of encodings
        for encoding in timestamp_encodings:
            self.data.insert(self.data.shape[1], self.data.shape[1], [encoding(x) for x in self.data.index])

        # Update data tensor
        self.tensor_data = torch.tensor(self.data.to_numpy(),dtype=torch.float32)

        # Update list of covariate columns
        self.cov_columns = list(set(self.data.columns) - set(self.pred_columns))

    def __len__(self) -> int:
        """
        Returns:
            Number of samples in the dataset
        """

        return max(0, self.tensor_data.shape[0] - self.context_length - 1)

    def __getitem__(self, idx: int) -> Tensor:
        """
        Args:
            idx: Index of the sample to return
        Returns:
            Tuple of context, covariates, and target Tensors
        """
        
        context = self.tensor_data[idx:idx + self.context_length, self.pred_columns].T.detach().clone()
        covariates = self.tensor_data[idx:idx + self.context_length, self.cov_columns].T.detach().clone()
        target = self.tensor_data[idx + self.context_length, self.pred_columns].detach().clone()
        return context, covariates, target