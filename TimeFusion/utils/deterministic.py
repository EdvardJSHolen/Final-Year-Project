# Library imports
import torch
import time
import os

# Module imports
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Callable, Optional, Dict
from tqdm import tqdm

# Relative imports
from utils.data import TimeFusionDataset
from utils.scaler import MeanScaler
from utils.early_stopper import EarlyStopper

class DeterministicForecaster(nn.Module):

    def __init__(
            self,
            input_size: int,
            output_size: int,
            hidden_size: int = 40,
            rnn_layers: int = 2,
            fc_layers: int = 2,
            scaling: bool = False,
            device: torch.device = torch.device("cpu"),
        ) -> None:
        
        # Init nn.Module base class
        super().__init__()

        assert fc_layers >= 1, "Need to have a least one linear layer after rnn!"

        self.scaling = scaling
        self.device = device


        if scaling:
            self.scaler = MeanScaler(
                device = device,
                min_scale = 0.01,
            )

        self.rnn = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = rnn_layers,
            batch_first = True,
            device = device
        )

        layers = []
        for _ in range(fc_layers-1):
            layers.append(nn.Linear(
                in_features = hidden_size,
                out_features = hidden_size,
                device = device
            ))
            layers.append(nn.ReLU())

        layers.append(
            nn.Linear(
                in_features = hidden_size,
                out_features = output_size,
                device = device
            )
        )

        self.fc = nn.Sequential(*layers)

    
    def forward(self, x: Tensor):
        """
        Args:
            x: Tensor of shape (batch size, time-series dim, context length)
        """

        x = x.permute((0,2,1))
        h, _  = self.rnn(x)
        x = self.fc(h[:,-1])

        return x
    

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

                # Zero gradients
                optimizer.zero_grad()

                # Make prediction
                predictions = self.forward(torch.concat((context,covariates),dim = 1))

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

                        # Make prediction
                        predictions = self.forward(torch.concat((context,covariates),dim = 1))

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
        batch_size: int = 64,
    ) -> Tensor:
        
        """
        Args:
            data: Dataset from which context data should be fetched
            indices: List of indices at which to predict
            prediction_length: Number of time-steps to predict
            batch_size: Maximum batch size of Tensors fed to predictor
        Output:
            Tensor of shape (len(idx),num time-series,prediction length)
        """

        # Set the network into evaluation mode
        self.train(False)
        
        samples = torch.empty((len(indices),len(data.pred_columns),prediction_length), dtype=torch.float32, device=self.device)
        for batch_idx in range(0,len(indices),batch_size):
            for pred_idx in range(prediction_length):

                # Get context Tensor at indices in batch
                context = torch.stack([data.get_sample_tensor(i + pred_idx) for i in indices[batch_idx:batch_idx+batch_size]])
                context = context.to(self.device)

                # Replace existing data into Tensor
                if pred_idx > 0:
                    context[:,data.pred_columns,-pred_idx:] = samples[batch_idx:batch_idx+batch_size,:,max(0,pred_idx - context.shape[2]):pred_idx]

                # Scale data
                if self.scaling:
                    context = self.scaler(context)

                # Predict future values
                predictions = self.forward(context)

                if self.scaling:
                    predictions = self.scaler.unscale(predictions)

                # Store denoised samples
                samples[batch_idx:batch_idx+batch_size,:,pred_idx] = predictions

        return samples