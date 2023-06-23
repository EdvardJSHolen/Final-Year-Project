def main():   
    import time
    import sys
    import os
    import torch
    import json
    import pandas as pd
    import numpy as np
    from torch.utils.data import DataLoader

    # Set path to fix relative imports
    sys.path.append("..")
    from utils.data import TimeFusionDataset
    from timefusion import TimeFusion
    from utils.early_stopper import EarlyStopper
    from training_scripts.results.performance import performance
    
    
    # Get environment variables
    process_id = int(os.environ["PBS_ARRAY_INDEX"])
    num_processes = int(os.environ["NUM_PROCESSES"])
    config_path = os.environ["CONFIG_PATH"]
    prediction_length = 24

    print(f"Process {process_id} of {num_processes} started.")

    # Get configurations
    configs = json.load(open(config_path,"r"))
    configs = np.array_split(configs,num_processes)[process_id]

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Import dataset
    train_data = pd.read_csv("../../datasets/electricity/train.csv", index_col="date")
    val_data = pd.read_csv("../../datasets/electricity/val.csv", index_col="date")

    # Normalize the signal power of each column
    stds = train_data.std()
    train_data /= stds
    val_data /= stds
    
    # Concatenate to get the full data
    full_data = pd.concat([train_data,val_data])

    # Function to create datasets and dataloaders
    def get_data_loader(data, context_length):

        dataset = TimeFusionDataset(
            data = data,
            context_length = context_length,
        )
        dataset.add_timestamp_encodings()

        dataloader = DataLoader(
            dataset = dataset,
            shuffle = True,
            num_workers = 3,
            batch_size = 128,
        )

        return dataloader, dataset


    # Dataframe to store results
    results = pd.DataFrame(
        columns = [
            "trial_number",
            "trial_start",
            "trial_end",
            *list(configs[0].keys()),
            "anchor_strength",
            "validation_mse",
            "validation_mae",
            "validation_mdae",
            "validation_crps_sum",
            "validation_variogram"
        ]
    )

    # Iterate over selected parameters
    for parameters in configs:    
        
        # Create datasets and loaders
        train_loader, train_dataset = get_data_loader(train_data, parameters["context_length"]*prediction_length)
        val_loader, val_dataset = get_data_loader(val_data, parameters["context_length"]*prediction_length)

        for trial_number in range(5):
            print(f"Trial number: {trial_number}")

            # Time at start of training
            trial_start = time.time()
            

            predictor = TimeFusion(
                input_size = train_dataset.data.shape[1],
                output_size = train_data.shape[1],
                rnn_layers = parameters["rnn_layers"],
                rnn_hidden = parameters["rnn_hidden"],
                residual_layers = parameters["residual_layers"],
                residual_hidden = parameters["residual_hidden"],
                dropout = parameters["dropout"],
                scaling = parameters["mean_scaler"],
                residual_scaler = parameters["residual_scaler"],
                device = device,
            )

            optimizer = torch.optim.Adam(params=predictor.parameters(), lr=parameters["learning_rate"])
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.001, total_iters=200)

            predictor.train_network(
                train_loader = train_loader,
                epochs=200,
                val_loader = val_loader,
                optimizer = optimizer,
                lr_scheduler= lr_scheduler,
                early_stopper=EarlyStopper(patience=200),
                disable_progress_bar = True,
            )

            # Anchors
            last_idx = val_dataset.tensor_data.shape[0] - prediction_length - parameters["context_length"]*prediction_length
            indices = list(range(last_idx, last_idx - prediction_length*14, -prediction_length))
            max_anchors = []
            min_anchors = []
            for idx in indices:
                off_idx = idx + parameters["context_length"]*prediction_length
                max_values = full_data.loc[full_data.index < val_data.index[off_idx]].max(axis=0)
                min_values = full_data.loc[full_data.index < val_data.index[off_idx]].min(axis=0)
                mean_values = full_data.loc[full_data.index < val_data.index[off_idx]].mean(axis=0)
                max_anchors.append(1.1*torch.tensor(max_values,dtype=torch.float32,device=device) - 0.1*torch.tensor(mean_values,dtype=torch.float32,device=device))
                min_anchors.append(1.1*torch.tensor(min_values,dtype=torch.float32,device=device) - 0.1*torch.tensor(mean_values,dtype=torch.float32,device=device))

            max_anchors = torch.stack(max_anchors)
            min_anchors = torch.stack(min_anchors)
            anchors = torch.stack([min_anchors,max_anchors],dim = -1).unsqueeze(2).repeat((1,1,prediction_length,1))

            # Get validation and test results and store in pandas dataframe
            for anchor_strength in [0,0.003,0.01,0.03]:
                print(f"Anchor strength: {anchor_strength}")

                # Validation
                val_metrics = performance(
                    predictor = predictor,
                    data = val_dataset,
                    indices = indices,
                    anchors=anchors,
                    anchor_strength=anchor_strength,
                    prediction_length=prediction_length,
                    parameters=parameters,
                )


                # Store data in dataframe
                results.loc[results.shape[0]] = {
                    **parameters,
                    "trial_number" : trial_number,
                    "trial_start" : trial_start,
                    "trial_end" : time.time(),
                    "anchor_strength": anchor_strength,
                    "validation_mse": val_metrics["mse"],
                    "validation_mae": val_metrics["mae"],
                    "validation_mdae":val_metrics["mdae"] ,
                    "validation_crps_sum": val_metrics["crps_sum"],
                    "validation_variogram": val_metrics["variogram_score"]
                }

            # Save results in csv file
            results.to_csv(f"results/new_electricity/{process_id}.csv", index=False)
            

if __name__ == "__main__":
    main()