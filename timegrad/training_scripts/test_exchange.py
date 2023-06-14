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
    from pts.model.time_grad import TimeGradEstimator
    from pts import Trainer
    from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error

    sys.path.append("../..")
    from timefusion.utils import metrics
    
    # Get environment variables
    config_path = os.environ["CONFIG_PATH"]
    prediction_length = 30

    # Get configurations
    parameters = json.load(open(config_path,"r"))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Import dataset
    train_data = pd.read_csv("../../datasets/exchange/train.csv")
    val_data = pd.read_csv("../../datasets/exchange/val.csv")
    test_data = pd.read_csv("../../datasets/exchange/test.csv")

    # Normalize the signal power of each column
    stds = train_data.std()
    train_data /= stds
    val_data /= stds
    test_data /= stds
    
    dates = pd.date_range(start="1970-01-01",periods = len(train_data) + len(val_data) + len(test_data), freq = "D")

    # Convert data into a glounts ListDataset
    def get_dataset(df: pd.DataFrame,  date, freq: str = "D", indices: List[int] = [-1]) -> ListDataset:
        return ListDataset(
            [
                {
                    "start": date, # Dummy date
                    "target": df.values[:i].T,
                }
                for i in indices
            ],
            freq=freq,
            one_dim_target=False
        )

    train_dataset = get_dataset(train_data,date = dates[0])
    val_dataset_14 = get_dataset(val_data,date = dates[len(train_data)], indices=list(range(val_data.shape[0], val_data.shape[0] - 14*prediction_length, -prediction_length)))
    test_dataset_14 = get_dataset(test_data,date = dates[len(train_data) + len(val_data)], indices=list(range(test_data.shape[0], test_data.shape[0] - 14*prediction_length, -prediction_length)))

    # Dataframe to store results
    results = pd.DataFrame(
        columns = [
            "trial_number",
            "trial_start",
            "trial_end",
            *list(parameters.keys()),
            "validation_mse",
            "validation_mae",
            "validation_mdae",
            "validation_crps_sum",
            "validation_variogram",
            "test_mse",
            "test_mae",
            "test_mdae",
            "test_crps_sum",
            "test_variogram"
        ]
    )
    
        
    for trial_number in range(5):
        print(f"Trial number: {trial_number}")

        # Time at start of training
        trial_start = time.time()

        estimator = TimeGradEstimator(
            target_dim=train_data.shape[1],
            prediction_length=prediction_length,
            context_length=parameters["context_length"]*prediction_length,
            input_size=38,
            freq="D",
            scaling=parameters["scaling"],
            diff_steps=parameters["diff_steps"],
            beta_schedule=parameters["beta_schedule"],
            num_cells = parameters["num_cells"],
            dropout_rate = parameters["dropout"],
            num_layers = parameters["num_layers"],
            residual_layers = parameters["residual_layers"],
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
        val_mse = mean_squared_error(realisations.flatten(), mean_predictions.flatten())
        val_mae = mean_absolute_error(realisations.flatten(), mean_predictions.flatten())
        val_mdae = median_absolute_error(realisations.flatten(), mean_predictions.flatten())

        # CRPS_sum and Variogram_score
        val_crps_sum = np.mean([metrics.crps_sum(samples[i], realisations[i]) for i in range(realisations.shape[0])])
        val_variogram_score = np.mean([metrics.variogram_score(samples[i], realisations[i], weights="local", window_size=2) for i in range(realisations.shape[0])])

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_dataset_14,
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
        test_mse = mean_squared_error(realisations.flatten(), mean_predictions.flatten())
        test_mae = mean_absolute_error(realisations.flatten(), mean_predictions.flatten())
        test_mdae = median_absolute_error(realisations.flatten(), mean_predictions.flatten())

        # CRPS_sum and Variogram_score
        test_crps_sum = np.mean([metrics.crps_sum(samples[i], realisations[i]) for i in range(realisations.shape[0])])
        test_variogram_score = np.mean([metrics.variogram_score(samples[i], realisations[i], weights="local", window_size=2) for i in range(realisations.shape[0])])

        print(f"Storing results {trial_number}")
        # Store data in dataframe
        results.loc[results.shape[0]] = {
            **parameters,
            "trial_number" : trial_number,
            "trial_start" : trial_start,
            "trial_end" : time.time(),
            "validation_mse": val_mse,
            "validation_mae": val_mae,
            "validation_mdae":val_mdae ,
            "validation_crps_sum": val_crps_sum,
            "validation_variogram": val_variogram_score,
            "test_mse": test_mse,
            "test_mae": test_mae,
            "test_mdae":test_mdae ,
            "test_crps_sum": test_crps_sum,
            "test_variogram": test_variogram_score
        }

        # Save results in csv file
        results.to_csv(f"results/test_exchange.csv", index=False)
        

if __name__ == "__main__":
    main()