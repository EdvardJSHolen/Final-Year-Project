import torch

from torch import Tensor, nn

class MeanScaler(nn.Module):
    def __init__(
            self, 
            min_scale: float = 1e-3,
            device: torch.device = torch.device("cpu"),
        ):

        super().__init__()

        self.min_scale = min_scale
        self.device = device
        self.scales = None

    def forward(self, x: Tensor, update_scales: bool = True):
        """
        Args:
            x: Tensor of any shape
            update_scales: Whether to update the stored scales or just scale the input based on the stored scales
        Returns:
            x: Input Tensor scaled by mean absolute values
        """

        if update_scales:

            means = torch.mean(x.abs(), dim = -1)
            self.scales = torch.maximum(means, torch.full(means.shape, self.min_scale, device = self.device))

        # Scale data
        if x.dim() == self.scales.dim():
            x /= self.scales
        else:
            x /= self.scales.unsqueeze(-1)

        return x
    
    def unscale(self, x: Tensor):
        """
        Args:
            x: Tensor of any shape compatible with stored scales
        Returns:
            x: Input Tensor scaled by stored scales
        """
        if x.dim == self.scales.dim:
            x *= self.scales
        else:
            x *= self.scales.unsqueeze(-1)

        return x