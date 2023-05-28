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
    
    # Check if we should kill this process
    if os.path.isfile("results/exchange/stop.txt"):
        exit()
    
    # Get environment variables
    process_id = int(os.environ["PBS_ARRAY_INDEX"])
    num_processes = int(os.environ["NUM_PROCESSES"])
    config_path = os.environ["CONFIG_PATH"]
    prediction_length = 30

    print(f"Process {process_id} of {num_processes} started.")

    # Get configurations
    configs = json.load(open(config_path,"r"))
    configs = np.array_split(configs,num_processes)[process_id]

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Import dataset
    train_data = pd.read_csv("../../datasets/exchange/train.csv", index_col="LocalTime")
    val_data = pd.read_csv("../../datasets/exchange/val.csv", index_col="LocalTime")

    # Normalize the signal power of each column
    stds = train_data.std()
    train_data /= stds
    val_data /= stds

    # Function to create datasets and dataloaders
    def get_data_loader(data, context_length):

        dataset = TimeFusionDataset(
            data = data,
            context_length = context_length,
        )

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
                device = device,
                residual_scaler = parameters["residual_scaler"]
            )

            optimizer = torch.optim.Adam(params=predictor.parameters(), lr=parameters["learning_rate"], weight_decay=parameters["weight_decay"])
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=100)

            predictor.train_network(
                train_loader = train_loader,
                epochs=200,
                val_loader = val_loader,
                optimizer = optimizer,
                lr_scheduler= lr_scheduler,
                early_stopper=EarlyStopper(patience=20),
                disable_progress_bar = True,
            )

            # Anchors
            max_anchors = 1.1*torch.tensor(train_data.values.max(axis=0),dtype=torch.float32,device=device) - 0.1*torch.tensor(train_data.values.mean(axis=0),dtype=torch.float32,device=device)
            min_anchors = 1.1*torch.tensor(train_data.values.min(axis=0),dtype=torch.float32,device=device) - 0.1*torch.tensor(train_data.values.mean(axis=0),dtype=torch.float32,device=device)
            anchors = torch.stack([min_anchors,max_anchors],dim = -1).unsqueeze(0).unsqueeze(2).repeat((14,1,prediction_length,1))

            # Get validation and test results and store in pandas dataframe
            for anchor_strength in [0,0.003,0.01,0.03]:
                print(f"Anchor strength: {anchor_strength}")
                 # Check if we should kill this process
                if os.path.isfile("results/exchange/stop.txt"):
                    exit()

                # Validation
                last_idx = val_dataset.tensor_data.shape[0] - prediction_length - parameters["context_length"]*prediction_length
                indices = list(range(last_idx, last_idx - prediction_length*14, -prediction_length))
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
            results.to_csv(f"results/exchange/{process_id}.csv", index=False)
            
            # Check if we should kill this process
            if os.path.isfile("results/exchange/stop.txt"):
                exit()

if __name__ == "__main__":
    main()