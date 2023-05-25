# Library imports
import torch
import os
import time

# Module imports
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Callable, Optional, Dict, Iterable
from tqdm import tqdm

# Relative imports
from epsilon_theta import EpsilonTheta
from utils.scaler import MeanScaler
from utils.diffusion import Diffuser
from utils.early_stopper import EarlyStopper
from utils.data import TimeFusionDataset

class TimeFusion(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        rnn_layers: int = 2,
        rnn_hidden: int = 40,
        autoencoder_layers: int = 1,
        autoencoder_latent: int = 40,
        activation_fn: nn.Module = nn.ReLU(),
        scaling: bool = False,
        diff_steps: int = 100,
        betas: List[float] = None,
        device: torch.device = torch.device("cpu")
    ):
        
        super().__init__()

        self.scaling = scaling
        self.diff_steps = diff_steps
        self.device = device

        if scaling:
            self.scaler = MeanScaler(
                min_scale = 0.01,
                device = device
            )

        self.diffuser = Diffuser(
            diff_steps = diff_steps,
            betas = betas,
            device = device
        )
        
        self.epsilon_theta = EpsilonTheta(
            input_size = input_size,
            output_size = output_size,
            rnn_layers = rnn_layers,
            rnn_hidden = rnn_hidden,
            autoencoder_layers = autoencoder_layers,
            autoencoder_latent = autoencoder_latent,
            activation_fn = activation_fn,
            diff_steps = diff_steps,
            device = device
        )

    def forward(self, x: Tensor, n: Tensor, context: Tensor = None, h: Tensor = None) -> Tensor:
        return self.epsilon_theta(x, n, context, h)

    def train_network(self,
        epochs: int,
        train_loader: DataLoader, 
        val_loader: Optional[DataLoader] = None,
        val_metrics: Dict[str,Callable] = {},
        loss_function: Callable = nn.MSELoss(),
        optimizer: optim.Optimizer = None,
        lr_scheduler: _LRScheduler = None,
        early_stopper: EarlyStopper = None,
        save_weights: bool = False,
        weight_folder: str = "weights",
        restore_weights: bool = True
    ):
        # Set the network into training mode
        self.train(True)

        if optimizer is None:
            optimizer = optim.Adam(params = self.parameters(), lr = 1e-3)

        # Set default validation metrics
        val_metrics = val_metrics | {"val_loss": loss_function}
            
        for epoch in range(1,epochs+1):

            pbar = tqdm(train_loader, unit = "batch", desc = f"Epoch: {epoch}/{epochs}")

            running_loss = 0
            for i, data in enumerate(pbar,1):

                context, covariates, target = data
                context = context.to(self.device)
                covariates = covariates.to(self.device)
                target = target.to(self.device)

                if self.scaling:
                    context = self.scaler(context)
                    target = self.scaler(target, update_scales = False)

                # Diffuse data
                x, target, n = self.diffuser.diffuse(target)

                # Zero gradients
                optimizer.zero_grad()

                # Make prediction
                predictions, _ = self.forward(x, n, torch.concat((context,covariates),dim = 1))

                # Calculate loss and update weights
                loss = loss_function(predictions,target)
                loss.backward()
                optimizer.step()

                # Print training statistics
                running_loss += loss.item()
                average_loss = running_loss / i
                pbar.set_postfix({"Training loss": f"{average_loss:.4f}"})

            if lr_scheduler:
                lr_scheduler.step()

            if val_loader is not None:
                with torch.no_grad():
                    running_loss = {key:0 for key in val_metrics.keys()}
                    for data in val_loader:
                        context, covariates, target = data
                        context = context.to(self.device)
                        covariates = covariates.to(self.device)
                        target = target.to(self.device)

                        if self.scaling:
                            context = self.scaler(context)
                            target = self.scaler(target, update_scales = False)

                        # Diffuse data
                        x, target, n = self.diffuser.diffuse(target)

                        # Make prediction
                        predictions, _ = self.forward(x, n, torch.concat((context,covariates),dim = 1))

                        # Calculate metrics
                        for key, metric_func in val_metrics.items():
                            running_loss[key] += metric_func(predictions,target).item() 

                    stat_string = ""
                    for metric, value in running_loss.items():
                        stat_string += f"{metric}: {value / len(val_loader):.4f} , "
                    print(stat_string)

            if not early_stopper is None:
                if early_stopper.early_stop(model = self, validation_loss = running_loss["val_loss"] if val_loader is not None else average_loss):
                    if restore_weights:
                        self.load_state_dict(early_stopper.best_weights)
                    break

        if save_weights:
            if (weight_folder != "") and (not os.path.exists(weight_folder)):
                os.makedirs(weight_folder)
            weight_path = weight_folder + "/" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
            torch.save(self.state_dict(), weight_path)
        
    @torch.no_grad()
    def sample(
        self,
        data: TimeFusionDataset,
        indices: List[int],
        prediction_length: int,
        num_samples: int = 1,
        batch_size: int = 64,
    ):

        # Enter evaluation mode
        self.train(False)
        
        samples = torch.empty((len(indices), num_samples,len(data.pred_columns),prediction_length), dtype = torch.float32, device = self.device)

        for i, idx in enumerate(indices):
            for pred_idx in range(prediction_length):
                for batch_idx in range(0, num_samples, batch_size):

                    # Get context Tensor at index
                    context, covariates = data.get_sample_tensor(idx + pred_idx)
                    context = context.to(self.device)
                    covariates = covariates.to(self.device)

                    # Repeat context and covariates to give correct batch size
                    context = context.unsqueeze(0).repeat(min(num_samples - batch_idx,batch_size),1,1)
                    covariates = covariates.unsqueeze(0).repeat(min(num_samples - batch_idx,batch_size),1,1)

                    # Replace existing data into Tensor
                    if pred_idx > 0:
                        context[...,-pred_idx:] = samples[i,batch_idx:batch_idx+context.shape[0],:,max(0,pred_idx - context.shape[2]):pred_idx]

                    if self.scaling:
                        context = self.scaler(context)

                    # Sample initial white noise
                    x = self.diffuser.initial_noise(
                        shape = context.shape[:-1]
                    )

                    # Denoising loop
                    h = None
                    for n in range(self.diff_steps,0,-1):

                        epsilon, h = self.forward(x, torch.full(context.shape[:1],n), torch.concat((context,covariates), dim = 1), h)

                        x = self.diffuser.denoise(
                            x = x,
                            epsilon = epsilon,
                            n = n
                        )

                    if self.scaling:
                        x = torch.squeeze(self.scaler.unscale(x.unsqueeze(-1)))

                    # Store denoised samples
                    samples[i,batch_idx:batch_idx+context.shape[0],:,pred_idx] = x.detach().clone()

        return samples
