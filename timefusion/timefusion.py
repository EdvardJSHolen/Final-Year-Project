# Library imports
import torch
import os
import time

# Module imports
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Callable, Optional, Dict
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
        residual_layers: int = 2,
        residual_hidden: int = 100,
        residual_scaler: bool = False,
        dropout: float = 0.0,
        scaling: bool = False,
        diff_steps: int = 100,
        betas: List[float] = None,
        device: torch.device = torch.device("cpu"),
        **kwargs
    ):
        
        """
        Args:
            input_size: Number of input features, i.e. time-series dimension + convariates dimension
            output_size: Number of output features, i.e. time-series dimension
            rnn_layers: Number of RNN layers
            rnn_hidden: Size of RNN hidden state
            residual_layers: Number of residual layers
            residual_hidden: Size of hidden layer in residual layers
            residual_scaler: Whether to use a scaler or a linear layer at the beginning the residual layers
            scaling: Whether to scale data using a MeanScaler
            dropout: Dropout rate of RNN network
            diff_steps: Number of diffusion steps
            betas: List of beta values for each step of the diffusion process
            device: Device to use for computation
        """
        
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
            residual_layers = residual_layers,
            residual_hidden = residual_hidden,
            residual_scaler = residual_scaler,
            dropout=dropout,
            diff_steps = diff_steps,
            device = device,
            **kwargs
        )


    def forward(self, x: Tensor, n: Tensor, context: Tensor = None, h: Tensor = None) -> Tensor:
        return self.epsilon_theta(x, n, context, h)

    def train_network(self,
        train_loader: DataLoader, 
        epochs: int = 200,
        val_loader: Optional[DataLoader] = None,
        val_metrics: Dict[str,Callable] = {},
        loss_function: Callable = nn.MSELoss(),
        optimizer: optim.Optimizer = None,
        lr_scheduler: _LRScheduler = None,
        early_stopper: EarlyStopper = EarlyStopper(patience = 200),
        save_weights: bool = False,
        weight_folder: str = "weights",
        restore_weights: bool = True,
        disable_progress_bar: bool = False,
    ):
        """
        Args:
            train_loader: DataLoader for training data
            epochs: Number of epochs to train
            val_loader: DataLoader for validation data
            val_metrics: Dictionary of validation metrics
            loss_function: Loss function to use for training
            optimizer: Optimizer to use for training
            lr_scheduler: Learning rate scheduler to use for training
            early_stopper: Early stopper to use for training
            save_weights: Whether to save the weights of the model
            weight_folder: Folder to save the weights of the model
            restore_weights: Whether to restore the best weights of the model after training
            disable_progress_bar: Whether to disable the progress bar
        """

        # Set the network into training mode
        self.train(True)

        if optimizer is None:
            optimizer = optim.Adam(params = self.parameters(), lr = 1e-3)

        # Set default validation metrics
        val_metrics = val_metrics | {"val_loss": loss_function}
            
        for epoch in range(1,epochs+1):

            pbar = tqdm(train_loader, unit = "batch", desc = f"Epoch: {epoch}/{epochs}", disable=disable_progress_bar)

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
                    break

        if not early_stopper is None and restore_weights:
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
        indices: List[int],
        prediction_length: int,
        num_samples: int = 1,
        anchors: Tensor = None,
        anchor_strength: float = 0.01,
        **kwargs
    ):
        
        """
        Args:
            data: Dataset to take context and covariate data from
            indices: List of indices of input data to use as starting point for prediction
            prediction_length: Number of time-steps to predict into the future
            num_samples: Number of samples to draw at each index
            anchors: Anchors to use for sampling
            anchor_strength: Strength of anchoring in denoising process
        """

        # Enter evaluation mode
        self.train(False)

        if anchors is not None:
            anchors = anchors.to(self.device)
        
        samples = torch.empty((len(indices) * num_samples,len(data.pred_columns),prediction_length), dtype = torch.float32, device = self.device)

        for pred_idx in range(prediction_length):

            # Get context and covariates Tensors
            context = []
            covariates = []
            for idx in indices:
                tmp_context, tmp_covariates, _ = data[idx + pred_idx]
                context.append(tmp_context.unsqueeze(0).repeat(num_samples,1,1))
                covariates.append(tmp_covariates.unsqueeze(0).repeat(num_samples,1,1))                
            context = torch.concat(context, dim = 0).to(self.device)
            covariates = torch.concat(covariates, dim = 0).to(self.device)

            if anchors is not None:
                anchors_copy = anchors[...,pred_idx,:].unsqueeze(1).repeat(1,num_samples,1,1).flatten(start_dim=0,end_dim=1).detach().clone()

            # Replace existing data into Tensor
            if pred_idx > 0:
                context[...,-pred_idx:] = samples[...,max(0,pred_idx - context.shape[2]):pred_idx]

            if self.scaling:
                context = self.scaler(context)
                if anchors is not None:
                    anchors_copy = self.scaler(anchors_copy, update_scales = False)


            # Sample initial white noise
            x = self.diffuser.initial_noise(
                shape = context.shape[:-1]
            )

            # Denoising loop
            h = None
            for n in range(self.diff_steps,0,-1):

                epsilon, h = self.forward(x, torch.full(context.shape[:1],n,device=self.device), torch.concat((context,covariates), dim = 1), h)

                x = self.diffuser.denoise(
                    x = x,
                    epsilon = epsilon,
                    n = n,
                    anchors = anchors_copy if anchors is not None else None,
                    anchor_strength = anchor_strength
                )

            if self.scaling:
                x = self.scaler.unscale(x)

            # Store denoised samples
            samples[...,pred_idx] = x.detach().clone()

        return samples.reshape((len(indices), num_samples, len(data.pred_columns), prediction_length))
