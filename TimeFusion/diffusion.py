import torch
import random
import math
import numpy as np
from typing import List, Tuple
from torch import Tensor

# Module to diffuse data
class Diffuser():

    def __init__(
        self,
        prediction_length: int,
        diff_steps: int = 100,
        betas: List[float] = None,
        device: torch.device = torch.device("cpu") 
    ) -> None:
        self.prediction_length = prediction_length

        # Diffusion parameters
        self.diff_steps = diff_steps
        if betas is None:
            self.betas = torch.linspace(1e-4, 0.1, diff_steps, device=device)
        else:
            self.betas = torch.tensor(betas, dtype=torch.float32, device=device)

        self.alphas = 1.0 - self.betas
        #self.bar_alphas = torch.cumprod(self.alphas,dim=0)
        self.bar_alphas = torch.tensor(
            np.cumprod(self.alphas.cpu()),
            dtype=torch.float32,
            device=device
        )

        # Device to perform diffusion on
        self.device = device

    @torch.no_grad()
    def diffuse(self, tokens: Tensor) -> Tuple[Tensor,Tensor]:
        # Sample targets from N(0,I)
        targets = torch.empty(size=tokens[:,:,-self.prediction_length:,0].shape, device=self.device).normal_()

        # Diffuse data
        n = torch.randint(1, self.diff_steps + 1, size = tokens.shape[:1], device=self.device)
        tokens[:,:,-self.prediction_length:,-1] = (n / self.diff_steps).view(-1,1,1)#(2*n / self.diff_steps - 1).view(-1,1,1)
        tokens[:,:,-self.prediction_length:,0] = torch.sqrt(self.bar_alphas[n-1]).view((-1,1,1)) * tokens[:,:,-self.prediction_length:,0] + torch.sqrt(1 - self.bar_alphas[n - 1]).view((-1,1,1)) * targets

        alphas_out = self.alphas[n-1]
        bar_alphas_out = self.bar_alphas[n-1]

        return tokens, targets, alphas_out, bar_alphas_out

    @torch.no_grad()
    def denoise(self, tokens: Tensor, epsilon: Tensor, n: int) -> Tensor:
        """returns x^n"""
            
        assert 0 <= n <= self.diff_steps, "Requested diffusion step exceeds the defined diffusion step range"

        # Set diffusion index
        tokens[:,:,-self.prediction_length:,-1] = torch.full(tokens[:,:,-self.prediction_length:,-1].shape, n/self.diff_steps)#2*n / self.diff_steps - 1)
    
        if n == self.diff_steps:
            # Sample initial white noise
            z = torch.empty(size=tokens[:,:,-self.prediction_length:,0].shape,device=self.device).normal_()
            tokens[:,:,-self.prediction_length:,0] = z
            return tokens
        elif n > 1:
            z = torch.empty(size=tokens[:,:,-self.prediction_length:,0].shape,device=self.device).normal_()
        else:
            z = torch.zeros(size=tokens[:,:,-self.prediction_length:,0].shape,device=self.device)

        tokens[:,:,-self.prediction_length:,0] = (1/math.sqrt(self.alphas[n-1]))*(tokens[:,:,-self.prediction_length:,0] - (self.betas[n-1]/math.sqrt(1-self.bar_alphas[n-1]))*epsilon) + math.sqrt(self.betas[n-1])*z

        return tokens