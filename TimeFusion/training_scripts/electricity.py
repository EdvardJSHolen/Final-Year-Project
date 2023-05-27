def main():   
    import time
    import sys
    import os
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error
    import numpy as np
    import pandas as pd
    import random

    # Set path to fix relative imports
    sys.path.append("..")
    from utils import metrics
    from utils.data import TimeFusionDataset
    from timefusion import TimeFusion
    from utils.early_stopper import EarlyStopper
    from typing import Dict, Any

    # Get environment variables
    process_id = int(os.environ["PBS_ARRAY_INDEX"])
    num_processes = int(os.environ["NUM_PROCESSES"])
    num_trials = int(os.environ["NUM_TRIALS"])

    if not os.path.exists("results/electricity"):
        os.makedirs("results/electricity")

    print(f"Process {process_id} of {num_processes} started.")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Import dataset
    train_data = pd.read_csv("../../datasets/electricity/train.csv").set_index("date")
    test_data = pd.read_csv("../../datasets/electricity/test.csv").set_index("date")
    train_data = train_data.iloc[:,:30]
    test_data = test_data.iloc[:,:30]
    train_data.index = pd.to_datetime(train_data.index)
    test_data.index = pd.to_datetime(test_data.index)

    # Normalize std of data to 1
    std = train_data.std()
    train_data /= std
    test_data /= std

    def get_hyperparameters(ranges: Dict[str,Any], parameter_id: int) -> Dict[str,Any]:
        """
        Returns a dictionary of hyperparameters for a given parameter_id.
        """

        div_terms = list(np.cumprod([len(values) for _, values in ranges]))
        div_terms.insert(0,1)
        div_terms.pop(-1)

        return {ranges[i][0]:ranges[i][1][int(parameter_id // div) % len(ranges[i][1])] for i, div in enumerate(div_terms)}
    
    # Hyperparameter ranges
    ranges = [
        ("context_length",[24,48,96]),
        ("rnn_layers",[2,3,4]),
        ("rnn_hidden",[1/3,1/2,1,2]),
        ("residual_layers",[1,2,4]),
        ("residual_hidden",[1/3,1/2,1,2]),
        ("weight_decay",[0,1e-5,1e-4]),
        ("dropout",[0,0.1,0.01]),
        ("learning_rate",[1e-3,1e-4]),
    ]


    # Randomly select combinations to test
    random.seed(0)
    max_parameter_id = np.prod([len(values) for _, values in ranges])
    parameters_ids = random.sample(range(int(max_parameter_id)),num_trials)

    # Get the subset to be computed by this process
    selected_parameters = [get_hyperparameters(ranges,parameter_id) for parameter_id in np.array_split(parameters_ids,num_processes)[process_id]]

    # Dataframe to store results
    results = pd.DataFrame(
        columns = [
            "trial_number",
            "trial_start",
            "trial_end",
            "context_length",
            "rnn_layers",
            "rnn_hidden",
            "residual_layers",
            "residual_hidden",
            "weight_decay",
            "dropout",
            "learning_rate",
            "anchor_strength",
            "validation_mse",
            "validation_mae",
            "validation_mdae",
            "validation_crps_sum",
            "validation_variogram",
            "test_mse",
            "test_mae",
            "test_mdae",
            "test_crps_sum",
            "test_variogram",
        ]
    )

    prediction_length = 24

    # Iterate over selected parameters
    for parameters in selected_parameters:    
        
        # Create datasets
        train_dataset = TimeFusionDataset(
            data = train_data.iloc[:int(0.9*len(train_data))],
            context_length = parameters["context_length"],
        )
        train_dataset.add_timestamp_encodings()

        val_dataset = TimeFusionDataset(
            data = train_data.iloc[int(0.9*len(train_data)):],
            context_length = parameters["context_length"],
        )
        val_dataset.add_timestamp_encodings()

        test_dataset = TimeFusionDataset(
            data = test_data,
            context_length = parameters["context_length"],
        )
        test_dataset.add_timestamp_encodings()

        train_loader = DataLoader(
            dataset = train_dataset,
            shuffle = True,
            num_workers = 4,
            batch_size = 128,
        )

        val_loader = DataLoader(
            dataset = val_dataset,
            shuffle = True,
            num_workers = 4,
            batch_size = 128,
        )

        for trial_number in range(5):
            print(f"Trial number: {trial_number}")

            # Time at start of training
            trial_start = time.time()
            
            rnn_hidden = int(parameters["rnn_hidden"]*train_data.shape[1])
            residual_hidden = int((rnn_hidden + train_data.shape[1])*parameters["residual_hidden"])

            predictor = TimeFusion(
                input_size = train_dataset.data.shape[1],
                output_size = train_data.shape[1],
                rnn_layers = parameters["rnn_layers"],
                rnn_hidden = rnn_hidden,
                residual_layers = parameters["residual_layers"],
                residual_hidden = residual_hidden,
                dropout = parameters["dropout"],
                scaling = True,
                device = device
            )


            optimizer = torch.optim.Adam(params=predictor.parameters(), lr=parameters["learning_rate"], weight_decay=parameters["weight_decay"])
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=100)

            predictor.train_network(
                train_loader = train_loader,
                epochs=100,
                val_loader = val_loader,
                val_metrics= {
                    "Val MAE": nn.L1Loss(),
                },
                optimizer = optimizer,
                lr_scheduler= lr_scheduler,
                early_stopper=EarlyStopper(patience=20),
                disable_progress_bar = True,
            )


            # Get validation and test results and store in pandas dataframe
            for anchor_strength in [0,0.005,0.01,0.02,0.04]:
                print(f"Anchor strength: {anchor_strength}")

                # Anchors
                max_anchors = 1.1*torch.tensor(train_data.values.max(axis=0),dtype=torch.float32) - 0.1*torch.tensor(train_data.values.mean(axis=0),dtype=torch.float32)
                min_anchors = 1.1*torch.tensor(train_data.values.min(axis=0),dtype=torch.float32) - 0.1*torch.tensor(train_data.values.mean(axis=0),dtype=torch.float32)
                anchors = torch.stack([min_anchors,max_anchors],dim = -1).unsqueeze(0).unsqueeze(2).repeat((14,1,prediction_length,1))

                # Anchors

                # Validation
                last_idx = val_dataset.tensor_data.shape[0] - prediction_length - parameters["context_length"]
                indices = list(range(last_idx, last_idx - 24*14, -24))

                samples = predictor.sample(
                    data = val_dataset,
                    indices = indices,
                    prediction_length = prediction_length,
                    num_samples = 128,
                    batch_size = 128,
                    anchors = anchors,
                    anchor_strength = anchor_strength,
                )
                samples = samples.cpu().numpy()

                realisations = []
                for idx in indices:
                    realisations.append(
                        val_dataset.tensor_data[parameters["context_length"] + idx:parameters["context_length"] + idx + prediction_length, val_dataset.pred_columns].T
                    )
                realisations = np.stack(realisations)

                # Calculate metrics
                mean_predictions = samples.mean(axis=1)
                val_mse = mean_squared_error(realisations.flatten(), mean_predictions.flatten())
                val_mae = mean_absolute_error(realisations.flatten(), mean_predictions.flatten())
                val_mdae = median_absolute_error(realisations.flatten(), mean_predictions.flatten())
                val_crps_sum = np.mean([metrics.crps_sum(samples[i], realisations[i]) for i in range(realisations.shape[0])])
                val_variogram_score = np.mean([metrics.variogram_score(samples[i], realisations[i], weights="local", window_size=3) for i in range(realisations.shape[0])])
                

                # Test
                last_idx = test_dataset.tensor_data.shape[0] - prediction_length - parameters["context_length"]
                indices = list(range(last_idx, last_idx - 24*14, -24))

                samples = predictor.sample(
                    data = test_dataset,
                    indices = indices,
                    prediction_length = prediction_length,
                    num_samples = 128,
                    batch_size = 128,
                    anchors = anchors,
                    anchor_strength = anchor_strength,
                )
                samples = samples.cpu().numpy()


                realisations = []
                for idx in indices:
                    realisations.append(
                        test_dataset.tensor_data[parameters["context_length"] + idx:parameters["context_length"] + idx + prediction_length,test_dataset.pred_columns].T
                    )
                realisations = np.stack(realisations)

                # Calculate metrics
                mean_predictions = samples.mean(axis=1)
                test_mse = mean_squared_error(realisations.flatten(), mean_predictions.flatten())
                test_mae = mean_absolute_error(realisations.flatten(), mean_predictions.flatten())
                test_mdae = median_absolute_error(realisations.flatten(), mean_predictions.flatten())
                test_crps_sum = np.mean([metrics.crps_sum(samples[i], realisations[i]) for i in range(realisations.shape[0])])
                test_variogram_score = np.mean([metrics.variogram_score(samples[i], realisations[i], weights="local", window_size=3) for i in range(realisations.shape[0])])

                # Store data in dataframe
                results.loc[results.shape[0]] = {
                    "trial_number" : trial_number,
                    "trial_start" : trial_start,
                    "trial_end" : trial_start,
                    "context_length": parameters["context_length"],
                    "rnn_layers": parameters["rnn_layers"],
                    "rnn_hidden": parameters["rnn_hidden"],
                    "residual_layers": parameters["residual_layers"],
                    "residual_hidden": parameters["residual_hidden"],
                    "weight_decay": parameters["weight_decay"],
                    "dropout": parameters["dropout"],
                    "learning_rate": parameters["learning_rate"],
                    "anchor_strength": anchor_strength,
                    "validation_mse": val_mse,
                    "validation_mae": val_mae,
                    "validation_mdae": val_mdae,
                    "validation_crps_sum": val_crps_sum,
                    "validation_variogram": val_variogram_score,
                    "test_mse": test_mse,
                    "test_mae": test_mae,
                    "test_mdae": test_mdae,
                    "test_crps_sum": test_crps_sum,
                    "test_variogram": test_variogram_score,
                }

            trial_end = time.time()

            # Save results in csv file
            results.to_csv(f"results/electricity/{process_id}.csv", index=False)

if __name__ == "__main__":
    main()