# Library imports
import torch
import math
import time
import os

# Module imports
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Callable, Optional, Dict, Tuple

# Relative imports
from data import TimeFusionDataset
from diffusion import Diffuser
from utils.modules import MeanScaler, EarlyStopper

class DiffusionEmbedding(nn.Module):

    def __init__(self, dim: int, proj_dim: int, device: torch.device, max_steps: int = 500):
        super().__init__()

        # Look up table for sine encodings
        step = torch.arange(max_steps, device = device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_steps, dim, device = device)
        pe[:, 0::2] = torch.sin(step * div_term)
        pe[:, 1::2] = torch.cos(step * div_term)
        self.register_buffer('pe', pe)

        # FC network
        self.projection1 = nn.Linear(dim, proj_dim,device=device)
        self.tanh1 = nn.Tanh()
        self.projection2 = nn.Linear(proj_dim, proj_dim,device=device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor with shape [batch_size] giving the diffusion step
        """
        x = self.pe[x]
        x = self.projection1(x)
        x = self.tanh1(x)
        x = self.projection2(x)
        return x
    

# class ResidualBlock(nn.Module):

#     def __init__(self, x_dim: int, hidden_dim: int, noise_dim: int, device: torch.device):
#         super().__init__()

#         total_dim = x_dim + hidden_dim + noise_dim

#         # FC network
#         self.projection1 = nn.Linear(total_dim, total_dim)
#         self.projection2 = nn.Linear(total_dim, x_dim)
#         self.relu = nn.ReLU()

#     def forward(self, x: Tensor, h: Tensor, n: Tensor) -> Tensor:

#         out = torch.concat([x,h,n],dim = 1)
#         out = self.projection1(out)
#         out = self.relu(out)
#         out = self.projection2(out)
#         out = self.relu(out + x)

#         return out


# Class accessible to end-user
class TimeFusion(nn.Module):

    def __init__(
            self,
            prediction_length: int,
            input_size: int,
            output_size: int,
            hidden_size: int = 40,
            recurrent_layers: int = 2,
            dropout: float = 0.1,
            residual_layers: int = 2,
            scaling: bool = False,
            diff_steps: int = 100,
            betas: List[float] = None,
            device: torch.device = torch.device("cpu"),
            **kwargs
        ) -> None:
        
        # Init nn.Module base class
        super().__init__()

        ### Set instance variables ###
        self.prediction_length = prediction_length
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_layers = recurrent_layers
        self.residual_layers = residual_layers
        self.scaling = scaling
        self.diff_steps = diff_steps
        self.device = device

        if scaling:
            self.scaler = MeanScaler(
                device = device,
                min_scale = 0.01,
                scaler_kwargs = kwargs.get("scaler_kwargs")
            )

        self.diff_embedding = DiffusionEmbedding(
            dim = 32,
            proj_dim = 32,
            device= device
        )

        self.diffuser = Diffuser(
            diff_steps = diff_steps,
            betas = betas,
            device = device
        )
        
        self.rnn = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = recurrent_layers,
            dropout = dropout,
            batch_first = True,
            device=device
        )
        
        self.linear1 = nn.Linear(
            in_features = output_size + hidden_size + 32,
            out_features = output_size + hidden_size + 32,
            device =device
        )

        self.linear2 = nn.Linear(
            in_features = output_size + hidden_size + 32,
            out_features = output_size + hidden_size + 32,
            device=device
        )

        self.linear3 = nn.Linear(
            in_features = output_size + hidden_size + 32,
            out_features = output_size,
            device=device
        )

        self.relu = nn.ReLU()

    
    def forward(self, context: Tensor, x: Tensor, n: Tensor):
        """
        Args:
            context: Tensor of shape (batch size, time-series dim, context length)
            x: Tensor of shape (batch size, time-series dim)
            n: Tensor of shape (batch size)
        """
        context = context.permute((0,2,1))
        h, _  = self.rnn(context)

        n = self.diff_embedding(n)

        x = self.linear1(torch.concat([x,h[:,-1],n],dim=1))
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        
        return x
    
    def partial_forward(self, x: Tensor, n: Tensor, context: Tensor = None, h: Tensor = None) -> Tuple[Tensor, Tensor]:

        assert (not context is None) or (not h is None), "Either context or hidden state of LSTM must be provided"

        if h is None:
            context = context.permute((0,2,1))
            h, _  = self.rnn(context)
        
        n = self.diff_embedding(n)
        x = self.linear1(torch.concat([x,h[:,-1],n],dim=1))
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)

        return x, h

    # Function to train TimeFusion network
    def train_network(self,
            train_loader: DataLoader, 
            epochs: int,
            val_loader: Optional[DataLoader] = None,
            val_metrics: Optional[Dict[str,Callable]] = None,
            loss_function: Callable = nn.MSELoss(),
            optimizer: optim.Optimizer = None,
            lr_scheduler: _LRScheduler = None,
            early_stopper: EarlyStopper = None,
            save_weights: bool = False,
            weight_folder: str = ""
        ) -> None:
        """
        Args:
            train_loader: Generator which provides batched training data.
            epochs: The number of epochs to train for.
            train_loader: Generator which provides batched validation data.
            val_metrics: Name of metrics and callable functions to calculate them which measure the performance of the network.
            loss_function: Function to measure how well predictions match targets.
            optimizer: Optimizer used to update weights.
            lr_scheduler: Learning rate scheduler which modifies learning rate after each epoch.
        """

        # Set the network into training mode
        self.train(True)

        # Set default optimizer
        if optimizer is None:
            optimizer = optim.Adam(params = self.parameters(), lr = 1e-4)

        # Set default validation metrics
        val_metrics = val_metrics | {"val_loss": loss_function}
            

        for epoch in range(1, epochs + 1):

            running_loss = 0
            for i, data in enumerate(train_loader, start = 1):
                context, target = data
                context = context.to(self.device)
                target = target.to(self.device)

                if self.scaling:
                    context = self.scaler(context)
                    target = torch.squeeze(self.scaler(target.unsqueeze(-1), update_scales = False))

                # Diffuse data
                x, target, n = self.diffuser.diffuse(target)

                # Zero gradients
                optimizer.zero_grad()

                # Forward, loss calculation, backward, optimizer step
                predictions = self.forward(context, x, n)

                loss = loss_function(predictions,target)
                loss.backward()
                optimizer.step()

                # Print training statistics
                running_loss += loss.item()
                average_loss = running_loss / i
                stat_string = "|" + "="*(30*i // len(train_loader)) + " "*(30 - (30*i // len(train_loader))) + f"|  Batch: {i} / {len(train_loader)}, Epoch: {epoch} / {epochs}, Average Loss: {average_loss:.4f}"
                print("\u007F"*512,stat_string,end="\r")

            if lr_scheduler:
                lr_scheduler.step()

            if val_loader is not None:
                with torch.no_grad():
                    running_loss = {key:0 for key in val_metrics.keys()}
                    for data in val_loader:
                        context, target = data
                        context = context.to(self.device)
                        target = target.to(self.device)

                        
                        if self.scaling:
                            context = self.scaler(context)
                            target = torch.squeeze(self.scaler(target.unsqueeze(-1), update_scales = False))

                        # Diffuse data
                        x, target, n = self.diffuser.diffuse(target)

                        # Calculate prediction metrics
                        predictions = self.forward(context, x, n)
                        for key, metric_func in val_metrics.items():
                            running_loss[key] += metric_func(predictions,target).item() 
                    
                    for metric, value in running_loss.items():
                        stat_string += f", {metric}: {value / len(val_loader):.4f}"

                    print("\u007F"*512,stat_string)

                    if not early_stopper is None:
                        stop = early_stopper.early_stop(running_loss["val_loss"] / len(val_loader),self)
                        if stop:
                            self.load_state_dict(early_stopper.best_weights)
                            break

            else:
                if not early_stopper is None:
                        stop = early_stopper.early_stop(average_loss,self)
                        if stop:
                            self.load_state_dict(early_stopper.best_weights)
                            break

                # New line for printing statistics
                print()

        if not early_stopper is None:
            self.load_state_dict(early_stopper.best_weights)

        if save_weights:
            if (weight_folder != "") and (not os.path.exists(weight_folder)):
                os.makedirs(weight_folder)
            weight_path = weight_folder + "/" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
            torch.save(self.state_dict(), weight_path)
        


    @torch.no_grad()
    def sample(
        self,
        data: TimeFusionDataset,
        idx: int,
        num_samples: int = 1,
        batch_size: int = 64,
    ):
        
        """
        Args:
            data:
            idx: 
            num_samples: Number of samples to sample from the network.
            batch_size: The number of samples to process at the same time.

        Note:
            1. For optimal performance, num_samples should be divisible by batch size.
            2. The timestamps must be drawn from the same distribution as those from the training data for best performance
        """

        # Set the network into evaluation mode
        self.train(False)
        
        samples = torch.empty((num_samples,len(data.ts_columns),self.prediction_length), dtype=torch.float32, device=self.device)
        # Sample for the same length as prediction length
        for pred_idx in range(self.prediction_length):
            
            # Split computation into batches
            for batch_idx in range(0, num_samples, batch_size):

                # Get context Tensor at index
                context = data.get_sample_tensor(idx + pred_idx)

                # Repeat token to give correct batch size
                context = context.unsqueeze(0).repeat(min(num_samples - batch_idx,batch_size),1,1)

                # Replace existing data into Tensor
                if pred_idx > 0:
                    context[:,data.ts_columns,-pred_idx:] = samples[batch_idx:batch_idx+batch_size,:,:pred_idx]

                # Scale data
                if self.scaling:
                    context = self.scaler(context)

                # Sample initial white noise
                x = self.diffuser.initial_noise(
                    shape = tuple([min(num_samples - batch_idx,batch_size), len(data.ts_columns)])
                )

                # Denoising loop
                h = None
                for n in range(self.diff_steps,0,-1):

                    # Calculate predicted noise
                    epsilon, h = self.partial_forward(x = torch.clone(x), n = torch.full([min(num_samples - batch_idx,batch_size)],n), context = context, h = h)

                    # Calculate x_n
                    x = self.diffuser.denoise(
                        x = x,
                        epsilon = epsilon,
                        n = n
                    )

                if self.scaling:
                    x = torch.squeeze(self.scaler.unscale(x.unsqueeze(-1)))

                # Store denoised samples
                samples[batch_idx:batch_idx+batch_size,:,pred_idx] = x

        return samples


# TODO:
# 2. restore weights option