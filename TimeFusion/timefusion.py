# Library imports
import torch
import math
import time
import os
import copy

# Module imports
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Callable, Optional, Dict
from tqdm import tqdm

# Relative imports
from data import EnsembleDataset
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


class Predictor(nn.Module):

    def __init__(
            self,
            input_size: int,
            output_size: int,
            hidden_size: int,
            recurrent_layers: int,
            fc_layers: int,
            scaling: bool,
            diff_steps: int,
            betas: List[float],
            device: torch.device = torch.device("cpu"),
            **kwargs
        ) -> None:
        
        # Init nn.Module base class
        super().__init__()

        ### Set instance variables ###
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.scaling = scaling
        self.device = device

        self.diffuser = Diffuser(
            diff_steps = diff_steps,
            betas = betas,
            device = device
        )

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

        self.rnn = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = recurrent_layers,
            batch_first = True,
            device=device
        )

        self.linear_layers = nn.ModuleList()
        for i in range(fc_layers):
            self.linear_layers.append(
                nn.Linear(
                    in_features = output_size + hidden_size + 32 if i == 0 else 100,
                    out_features = 100 if i < fc_layers - 1 else output_size,
                    device=device
                )
            )

        self.relu = nn.ReLU()

    def forward(self, context: Tensor, x: Tensor, n: Tensor, h: Tensor = None) -> None:
        """
        Args:
            context: Tensor of shape (batch size, time-series dim, context length)
            x: Tensor of shape (batch size, time-series dim)
            n: Tensor of shape (batch size)
        """

        if h is None:
            context = context.permute((0,2,1))
            h, _  = self.rnn(context)

        n = self.diff_embedding(n)

        x = torch.concat([x,h[:,-1],n],dim=1)
        for i, layer in enumerate(self.linear_layers):
            x = layer(x)
            if i < len(self.linear_layers) - 1:
                x = self.relu(x)

        return x, h
    
    def train_network(self,
        epochs: int,
        train_loader: DataLoader, 
        val_metrics: Dict[str,Callable],
        loss_function: Callable,
        optimizer: optim.Optimizer,
        lr_scheduler: _LRScheduler,
        val_loader: Optional[DataLoader] = None,
        early_stopper: EarlyStopper = None,
    ):
        
        for epoch in range(1,epochs+1):

            running_loss = 0

            p_bar = tqdm(train_loader, desc ="Training")
            for i, data in enumerate(p_bar,start=1):
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
                p_bar.set_description(f"Epoch {epoch}, Training loss: {average_loss:.4f}")

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

                        # Predict on data
                        predictions = self.forward(context, x, n)

                        # Calculate prediction metrics
                        for key, metric_func in val_metrics.items():
                            running_loss[key] += metric_func(predictions,target).item() 

                    stat_string = f"Validation loss: {running_loss['val_loss'] / len(val_loader):.4f}"
                    for metric, value in running_loss.items():
                        stat_string += f", {metric}: {value / len(val_loader):.4f}"
                    #p_bar.set_description(stat_string)
                    print(stat_string)

            if not early_stopper is None:
                if stop := early_stopper.early_stop(model = self, validation_loss = running_loss["val_loss"] if val_loader is not None else average_loss):
                    self.load_state_dict(early_stopper.best_weights)
                    break


# Class accessible to end-user
class Ensemble(nn.Module):

    def __init__(
            self,
            ensemble_size: int,
            prediction_length: int,
            input_size: int,
            output_size: int,
            hidden_size: int,
            recurrent_layers: int,
            fc_layers: int,
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
        self.ensemble_size = ensemble_size
        self.scaling = scaling
        self.diff_steps = diff_steps
        self.device = device

        if scaling:
            self.scaler = MeanScaler(
                device = device,
                min_scale = 0.01,
                scaler_kwargs = kwargs.get("scaler_kwargs")
            )

        self.diffuser = Diffuser(
            diff_steps = diff_steps,
            betas = betas,
            device = device
        )

        self.predictors = nn.ModuleList()
        for _ in range(ensemble_size):
            self.predictors.append(
                Predictor(
                    input_size = input_size,
                    output_size = output_size,
                    hidden_size = hidden_size,
                    recurrent_layers = recurrent_layers,
                    fc_layers = fc_layers,
                    scaling = scaling,
                    diff_steps = diff_steps,
                    betas = betas,
                    device = device,
                    **kwargs
                )
            )
    
    def forward(self, context: Tensor, x: Tensor, n: Tensor, h: Tensor = None) -> Tensor:

        predictions, hidden_states = list(zip(*[predictor(context[i],x[i],n[i],h[i] if h is not None else None) for i, predictor in enumerate(self.predictors)]))

        predictions = torch.stack(predictions)
        hidden_states = torch.stack(hidden_states)
        
        return predictions, hidden_states


    def train_ensemble(self,
            epochs: int,
            train_loaders: List[DataLoader], 
            val_loaders: Optional[List[DataLoader]] = None,
            val_metrics: Optional[Dict[str,Callable]] = None,
            loss_function: Callable = nn.MSELoss(),
            optimizers: List[optim.Optimizer] = None,
            lr_scheduler: List[_LRScheduler] = None,
            early_stopper: EarlyStopper = None,
            save_weights: bool = False,
            weight_folder: str = "",
            reuse_weights: bool = True
        ) -> None:

        self.train(True)

        if optimizers is None:
            optimizer = [optim.Adam(params = predictor.parameters(), lr = 1e-4) for predictor in self.predictors]

        # Default validation metrics
        val_metrics = val_metrics | {"val_loss": loss_function}
            

        for pred_num, predictor in enumerate(self.predictors):
            print(f"Training predictor {pred_num + 1}")

            # Reset early stopper
            if not early_stopper is None:
                early_stopper.reset()

            # Reuse weigths
            if reuse_weights:
                predictor.load_state_dict(copy.deepcopy(self.predictors[0].state_dict()))

            # Train predictor
            predictor.train_network(
                epochs = epochs,
                train_loader = train_loaders[pred_num],
                val_loader = val_loaders[pred_num] if val_loaders is not None else None,
                val_metrics = val_metrics,
                loss_function = loss_function,
                optimizer = optimizers[pred_num],
                lr_scheduler = lr_scheduler[pred_num] if lr_scheduler is not None else None,
                early_stopper = early_stopper,
            )

        # Save weights
        if save_weights:
            print("Saving weights")
            if (weight_folder != "") and (not os.path.exists(weight_folder)):
                os.makedirs(weight_folder)
            weight_path = weight_folder + "/" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
            torch.save(self.state_dict(), weight_path)

        print("Finished training ensemble")


    @torch.no_grad()
    def sample(
        self,
        data: EnsembleDataset,
        indices: List[int],
        coefficients: List[float] = None, 
        num_samples: int = 1,
        batch_size: int = 64,
    ):
        
        if coefficients is None:
            coefficients = [0 for _ in range(self.diff_steps)]

        # Set the network into evaluation mode
        self.train(False)
        
        samples = torch.empty((len(indices),self.ensemble_size,num_samples, len(data.ts_columns),self.prediction_length), dtype = torch.float32, device = self.device)
        for i, idx in enumerate(indices):
            for pred_idx in range(self.prediction_length):
                
                # Split computation into batches
                for batch_idx in range(0, num_samples, batch_size):

                    # Get context Tensor at index
                    context = data.get_sample_tensor(idx + pred_idx)
                    context = context.to(self.device)

                    # Repeat token to give correct batch size
                    context = context.unsqueeze(0).unsqueeze(0).repeat(self.ensemble_size, min(num_samples - batch_idx,batch_size),1,1)

                    # Replace existing data into Tensor
                    if pred_idx > 0:
                        context[...,data.ts_columns,-pred_idx:] = torch.clone(samples[i,:,batch_idx:batch_idx+batch_size,:,max(0,pred_idx - context.shape[2]):pred_idx])

                    # Context parts for each predictor
                    context_divisions = [
                        torch.clone(context[j,:,:,j::self.ensemble_size]) for j in reversed(range(self.ensemble_size))
                    ]
                    context_divisions = torch.stack(context_divisions)

                    # Scale data
                    if self.scaling:
                        context_divisions = self.scaler(context_divisions)

                    # Sample initial white noise
                    x = self.diffuser.initial_noise(
                        shape = (min(num_samples - batch_idx,batch_size), len(data.ts_columns))
                    )

                    x = x.unsqueeze(0).repeat((self.ensemble_size,1,1))

                    # Denoising loop
                    h = None
                    for n in range(self.diff_steps,0,-1):

                        # Calculate predicted noise
                        epsilon, h = self.forward(torch.clone(context_divisions), torch.clone(x), torch.full([self.ensemble_size,min(num_samples - batch_idx,batch_size)],n), h)

                
                        # Calculate x_n
                        x = self.diffuser.denoise(
                            x = x,
                            epsilon = epsilon,
                            n = n,
                            coefficient = coefficients[n-1]
                        )

                    # Unscale data
                    if self.scaling:
                        x = torch.squeeze(self.scaler.unscale(torch.clone(x)))
                    # Store denoised samples
                    samples[i,:,batch_idx:batch_idx+batch_size,:,pred_idx] = torch.clone(x)

        return samples  


# # TODO:
# # 2. restore weights option