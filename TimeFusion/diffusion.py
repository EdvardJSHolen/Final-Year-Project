import torch
import math
import numpy as np
from typing import List, Tuple
from torch import Tensor

# Module to diffuse data
class Diffuser():

    def __init__(self,
        diff_steps: int = 100,
        betas: List[float] = None,
        device: torch.device = torch.device("cpu") 
    ) -> None:

        # Diffusion parameters
        self.diff_steps = diff_steps
        if betas is None:
            self.betas = torch.linspace(1e-4, 0.1, diff_steps, device=device)
        else:
            self.betas = torch.tensor(betas, dtype = torch.float32, device = device)

        self.alphas = 1.0 - self.betas
        self.bar_alphas = torch.tensor(
            np.cumprod(self.alphas.cpu()),
            dtype = torch.float32,
            device = device
        )

        # Device to perform diffusion on
        self.device = device

    @torch.no_grad()
    def diffuse(self, x: Tensor) -> Tuple[Tensor,Tensor]:
        # Sample targets from N(0,I)
        targets = torch.empty(size = x.shape, device = self.device).normal_()

        # Diffuse data
        n = torch.randint(1, self.diff_steps + 1, size = [x.shape[0]], device=self.device)
        x = torch.sqrt(self.bar_alphas[n-1]).view((-1,1)) * x + torch.sqrt(1 - self.bar_alphas[n - 1]).view((-1,1)) * targets

        return x, targets, n

    @torch.no_grad()
    def denoise(self, x: Tensor, epsilon: Tensor, n: int) -> Tensor:
            
        assert 0 <= n <= self.diff_steps, "Requested diffusion step exceeds the defined diffusion step range"

        # Set diffusion index
    
        if n == self.diff_steps:
            # Sample initial white noise
            x = torch.empty(size = x.shape, device = self.device).normal_()
            return x
        elif n > 1:
            z = torch.empty(size = x.shape, device = self.device).normal_()
        else:
            z = torch.zeros(size = x.shape, device = self.device)

        x = (1/math.sqrt(self.alphas[n-1]))*(x - (self.betas[n-1]/math.sqrt(1-self.bar_alphas[n-1]))*epsilon) + math.sqrt(self.betas[n-1])*z

        return x