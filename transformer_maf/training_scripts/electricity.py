def main():   
    import time
    import sys
    import os
    import torch
    import json
    import pandas as pd
    import numpy as np

    from gluonts.evaluation.backtest import make_evaluation_predictions
    from gluonts.dataset.common import ListDataset
    from typing import List
    from pts.model.transformer_tempflow import TransformerTempFlowEstimator
    from pts import Trainer
    from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error

    sys.path.append("../..")
    from timefusion.utils import metrics
    
    # Check if we should kill this process
    if os.path.isfile("results/electricity/stop.txt"):
        exit()
    
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

    # Convert data into a glounts ListDataset
    def get_dataset(df: pd.DataFrame, freq: str = "h", indices: List[int] = [-1]) -> ListDataset:
        return ListDataset(
            [
                {
                    "start": df.index[0],
                    "target": df.values[:i].T,
                }
                for i in indices
            ],
            freq=freq,
            one_dim_target=False
        )

    train_dataset = get_dataset(train_data)
    val_dataset_14 = get_dataset(val_data, indices=list(range(val_data.shape[0], val_data.shape[0] - 14*prediction_length, -prediction_length)))


    # Dataframe to store results
    results = pd.DataFrame(
        columns = [
            "trial_number",
            "trial_start",
            "trial_end",
            *list(configs[0].keys()),
            "validation_mse",
            "validation_mae",
            "validation_mdae",
            "validation_crps_sum",
            "validation_variogram"
        ]
    )

    # Iterate over selected parameters
    for parameters in configs:    
        
        for trial_number in range(5):
            print(f"Trial number: {trial_number}")
            
            try:

                # Time at start of training
                trial_start = time.time()

                estimator = TransformerTempFlowEstimator(
                    input_size=1280,
                    target_dim=train_data.shape[1],
                    prediction_length=prediction_length,
                    flow_type=parameters["flow_type"],
                    freq="h",
                    d_model=parameters["d_model"],
                    dim_feedforward_scale=parameters["dim_feedforward_scale"],
                    num_heads=parameters["num_heads"],
                    num_encoder_layers=parameters["num_encoder_layers"],
                    num_decoder_layers=parameters["num_decoder_layers"],
                    dropout_rate=parameters["dropout_rate"],
                    n_blocks=parameters["n_blocks"],
                    hidden_size=parameters["hidden_size"],
                    n_hidden=parameters["n_hidden"],
                    conditioning_length=parameters["conditioning_length"],
                    dequantize=parameters["dequantize"],
                    context_length=parameters["context_length"]*prediction_length,
                    trainer = Trainer(
                        device=device,
                        learning_rate=parameters["learning_rate"],
                        clip_gradient = parameters["gradient_clipping"]
                    )
                )

                predictor = estimator.train(training_data=train_dataset, num_workers=3)

                forecast_it, ts_it = make_evaluation_predictions(
                    dataset=val_dataset_14,
                    predictor=predictor,
                    num_samples=128
                )

                samples = list(forecast_it)
                realisations = list(ts_it)

                samples = torch.tensor([sample.samples for sample in samples]).permute(0,1,3,2)
                realisations = torch.tensor([real.values[-prediction_length:] for real in realisations]).permute(0,2,1)

                # Calculate metrics
                mean_predictions = samples.mean(axis=1)

                # MSE, MAE, MDAE
                mse = mean_squared_error(realisations.flatten(), mean_predictions.flatten())
                mae = mean_absolute_error(realisations.flatten(), mean_predictions.flatten())
                mdae = median_absolute_error(realisations.flatten(), mean_predictions.flatten())

                # CRPS_sum and Variogram_score
                crps_sum = np.mean([metrics.crps_sum(samples[i], realisations[i]) for i in range(realisations.shape[0])])
                variogram_score = np.mean([metrics.variogram_score(samples[i], realisations[i], weights="local", window_size=2) for i in range(realisations.shape[0])])

                # Store data in dataframe
                results.loc[results.shape[0]] = {
                    **parameters,
                    "trial_number" : trial_number,
                    "trial_start" : trial_start,
                    "trial_end" : time.time(),
                    "validation_mse": mse,
                    "validation_mae": mae,
                    "validation_mdae":mdae ,
                    "validation_crps_sum": crps_sum,
                    "validation_variogram": variogram_score
                }

                # Save results in csv file
                results.to_csv(f"results/electricity/{process_id}.csv", index=False)

            except:
                # Store data in dataframe
                results.loc[results.shape[0]] = {
                    **parameters,
                    "trial_number" : trial_number,
                    "trial_start" : trial_start,
                    "trial_end" : time.time(),
                    "validation_mse": np.nan,
                    "validation_mae": np.nan,
                    "validation_mdae": np.nan ,
                    "validation_crps_sum": np.nan,
                    "validation_variogram": np.nan
                }

                # Save results in csv file
                results.to_csv(f"results/electricity/{process_id}.csv", index=False)
            
            # Check if we should kill this process
            if os.path.isfile("results/electricity/stop.txt"):
                exit()

if __name__ == "__main__":
    main()