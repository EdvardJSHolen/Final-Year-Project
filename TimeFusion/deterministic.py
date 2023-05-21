# Library imports
import torch
import time
import os
import copy

# Module imports
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Callable, Optional, Dict

# Relative imports
from data import TimeFusionDataset
from utils.modules import MeanScaler, EarlyStopper

class DeterministicForecaster(nn.Module):

    def __init__(
            self,
            prediction_length: int,
            input_size: int,
            output_size: int,
            hidden_size: int = 40,
            fc_layers: int = 3,
            recurrent_layers: int = 2,
            dropout: float = 0.0,
            scaling: bool = False,
            device: torch.device = torch.device("cpu"),
            **kwargs
        ) -> None:
        
        # Init nn.Module base class
        super().__init__()

        ### Set instance variables ###
        self.prediction_length = prediction_length
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.fc_layers = fc_layers
        self.recurrent_layers = recurrent_layers
        self.scaling = scaling
        self.device = device

        assert fc_layers >= 1, "Need to have a least one linear layer after rnn!"

        if scaling:
            self.scaler = MeanScaler(
                device = device,
                min_scale = 0.01,
                scaler_kwargs = kwargs.get("scaler_kwargs",{})
            )

        self.rnn = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = recurrent_layers,
            batch_first = True,
            device = device
        )

        self.linear = nn.ModuleList()
        for i in range(fc_layers - 1):
            self.linear.append(
                nn.Linear(
                    in_features = hidden_size,
                    out_features = hidden_size,
                    device = device
                )
            )

        self.linear_out = nn.Linear(
            in_features = hidden_size,
            out_features = output_size,
            device = device
        )

        self.relu = nn.ReLU()

    
    def forward(self, x: Tensor):
        """
        Args:
            x: Tensor of shape (batch size, time-series dim, context length)
        """

        x = x.permute((0,2,1))
        h, _  = self.rnn(x)
        x = h[:,-1]

        for layer in self.linear:
            x = layer(x)
            x = self.relu(x)

        x = self.linear_out(x)

        return x
    

    # Function to train TimeFusion network
    def train_network(self,
            epochs: int,
            train_loader: DataLoader, 
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
            epochs: The number of epochs to train for.
            train_loader: Generator which provides batched training data.
            train_loader: Generator which provides batched validation data.
            val_metrics: Name of metrics and callable functions to calculate them which measure the performance of the network.
            loss_function: Function to measure how well predictions match targets.
            optimizer: Optimizer used to update weights.
            lr_scheduler: Learning rate scheduler which modifies learning rate after each epoch.
            early_stopper: EarlyStopper object which stops training early based on some requirements.
            save_weights: Whether or not to save the weights of the network when training finishes
            weight_folder: Path to folder where network weights should be stored
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
                    #target = torch.squeeze(self.scaler(target.unsqueeze(-1), update_scales = False))

                # Zero gradients
                optimizer.zero_grad()

                # Forward, loss calculation, backward, optimizer step
                predictions = self.forward(context)

                if self.scaling:
                    predictions = self.scaler.unscale(predictions.unsqueeze(-1)).squeeze()
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
                            #target = torch.squeeze(self.scaler(target.unsqueeze(-1), update_scales = False))

                        # Calculate prediction metrics
                        predictions = self.forward(context)
                        if self.scaling:
                            predictions = self.scaler.unscale(predictions.unsqueeze(-1)).squeeze()
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
        idx: List[int],
        batch_size: int = 64
    ) -> Tensor:
        
        """
        Args:
            data: Dataset from which context data should be fetched
            idx: List of indices at which to 
            batch_size: Maximum batch size of Tensors fed to predictor
        Output:
            Tensor of shape (len(idx),prediction length)
        """

        # Set the network into evaluation mode
        self.train(False)
        
        samples = torch.empty((len(idx),len(data.ts_columns),self.prediction_length), dtype=torch.float32, device=self.device)
        for batch_idx in range(0,len(idx),batch_size):
            for pred_idx in range(self.prediction_length):

                # Get context Tensor at indices in batch
                context = torch.stack([data.get_sample_tensor(i + pred_idx) for i in idx[batch_idx:batch_idx+batch_size]])
                context = context.to(self.device)

                # Replace existing data into Tensor
                if pred_idx > 0:
                    context[:,data.ts_columns,-pred_idx:] = samples[batch_idx:batch_idx+batch_size,:,max(0,pred_idx - context.shape[2]):pred_idx]

                # Scale data
                if self.scaling:
                    context = self.scaler(context)

                # Predict future values
                predictions = self.forward(context)

                if self.scaling:
                    predictions = self.scaler.unscale(predictions.unsqueeze(-1)).squeeze()

                # Store denoised samples
                samples[batch_idx:batch_idx+batch_size,:,pred_idx] = predictions

        return samples
