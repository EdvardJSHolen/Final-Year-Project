import math
import copy
import torch

from torch import nn, Tensor


class EarlyStopper:
    def __init__(self, patience: int = 1, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.min_validation_loss = math.inf
        self.counter = 0
        self.best_weights = None

    def early_stop(self, validation_loss, model) -> bool:

        if validation_loss > self.min_validation_loss - self.min_delta:
            self.counter += 1
        else:
            self.counter = 0

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.best_weights = copy.deepcopy(model.state_dict())

        if self.counter >= self.patience:
            return True
        
        return False


class MeanScaler(nn.Module):
    def __init__(
            self, 
            device: torch.device,
            min_scale: float = 1e-3
        ):

        super().__init__()

        self.device = device
        self.min_scale = min_scale
        self.scales = None

    def forward(self, x: Tensor):
        """
        Args:
            x: Tensor of shape (batch size, time-series dim, time-series length)
        Returns:
            x: Input Tensor scaled by mean absolute values
        """
        # Calculate mean absolute value of each time series
        means = torch.mean(x.abs(), dim=2)

        # Replace means which are too small
        self.scales = torch.maximum(means, torch.full(means.shape, self.min_scale, device = self.device))
        # Scale data
        x /= self.scales.unsqueeze(-1)

        return x
    
    def unscale(self, x: Tensor):
        """
        Args:
            x: Tensor of shape (batch size, time-series dim, time-series length)
        Returns:
            x: Input Tensor scaled by stored scales
        """

        # Scale data
        x *= self.scales

        return x