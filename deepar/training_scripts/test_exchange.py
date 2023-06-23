def main():   
    import time
    import sys
    import os
    import torch
    import json
    import pandas as pd
    import numpy as np

    from gluonts.dataset.pandas import PandasDataset
    from gluonts.dataset.split import split
    from gluonts.torch import DeepAREstimator
    from gluonts.evaluation.backtest import make_evaluation_predictions
    from gluonts.torch.distributions import StudentTOutput
    from gluonts.torch.distributions import NormalOutput
    from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error

    sys.path.append("../..")
    from timefusion.utils import metrics

    
    # Get environment variables
    config_path = os.environ["CONFIG_PATH"]
    prediction_length = 30
    freq = "D"


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
    
    dates = pd.date_range(start="1970-01-01",periods = len(train_data) + len(val_data) + len(test_data), freq = "D")
    
    train_data.index = dates[:len(train_data)]
    val_data.index = dates[len(train_data):len(train_data) + len(val_data)]
    test_data.index = dates[len(train_data) + len(val_data):]

    # Normalize the signal power of each column
    stds = train_data.std()
    train_data /= stds
    val_data /= stds
    test_data /= stds

    # Get training, validation and test dataset
    train_flat = train_data.stack().reset_index()
    train_flat.columns = ["date", "series", "value"]
    train_dataset = PandasDataset.from_long_dataframe(train_flat, target="value",item_id="series",timestamp="date",freq=freq)

    val_flat = val_data.stack().reset_index()
    val_flat.columns = ["date", "series", "value"]
    val_dataset = PandasDataset.from_long_dataframe(val_flat, target="value",item_id="series",timestamp="date",freq=freq)
    val_dataset_14 = [PandasDataset.from_long_dataframe(val_flat.iloc[:-prediction_length*i*train_data.shape[1]] if i != 0 else val_flat, target="value",item_id="series",timestamp="date",freq=freq) for i in range(14)]

    test_flat = test_data.stack().reset_index()
    test_flat.columns = ["date", "series", "value"]
    test_dataset_14 = [PandasDataset.from_long_dataframe(test_flat.iloc[:-prediction_length*i*train_data.shape[1]] if i != 0 else test_flat, target="value",item_id="series",timestamp="date",freq=freq) for i in range(14)]


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
            "test_variogram",
        ]
    )
 
        
    for trial_number in range(5):
        print(f"Trial number: {trial_number}")

        # Time at start of training
        trial_start = time.time()
            
        estimator = DeepAREstimator(
            prediction_length = prediction_length, 
            freq = freq,
            context_length = parameters["context_length"]*prediction_length,
            num_layers = parameters["num_layers"],
            hidden_size = parameters["num_cells"],
            lr = parameters["learning_rate"],
            dropout_rate = parameters["dropout_rate"],
            distr_output = StudentTOutput() if parameters["likelihood"] == "student-T" else NormalOutput(),
            trainer_kwargs={"max_epochs": parameters["epochs"]}
        )
            
            
        predictor = estimator.train(training_data = train_dataset,validation_data = val_dataset, num_workers = 3)
            
        # Validation data
        samples = []
        realisations = []

        for dataset in val_dataset_14:

            forecast_it, ts_it = make_evaluation_predictions(
                dataset=dataset,
                predictor=predictor,
                num_samples=128
            )

            samples.append(list(forecast_it))
            realisations.append(list(ts_it))
            
        # Convert to tensors
        samples = torch.tensor(np.array([[l.samples for l in s] for s in samples])).permute(0,2,1,3)
        realisations = torch.tensor([np.array(r).squeeze()[:,-prediction_length:] for r in realisations])


        # Calculate metrics
        mean_predictions = samples.mean(axis=1)

        # MSE, MAE, MDAE
        val_mse = mean_squared_error(realisations.flatten(), mean_predictions.flatten())
        val_mae = mean_absolute_error(realisations.flatten(), mean_predictions.flatten())
        val_mdae = median_absolute_error(realisations.flatten(), mean_predictions.flatten())

        # CRPS_sum and Variogram_score
        val_crps_sum = np.mean([metrics.crps_sum(samples[i], realisations[i]) for i in range(realisations.shape[0])])
        val_variogram_score = np.mean([metrics.variogram_score(samples[i], realisations[i], weights="local", window_size=2) for i in range(realisations.shape[0])])
            
        # Test data
        samples = []
        realisations = []

        for dataset in test_dataset_14:

            forecast_it, ts_it = make_evaluation_predictions(
                dataset=dataset,
                predictor=predictor,
                num_samples=128
            )

            samples.append(list(forecast_it))
            realisations.append(list(ts_it))
            
        # Convert to tensors
        samples = torch.tensor(np.array([[l.samples for l in s] for s in samples])).permute(0,2,1,3)
        realisations = torch.tensor([np.array(r).squeeze()[:,-prediction_length:] for r in realisations])


        # Calculate metrics
        mean_predictions = samples.mean(axis=1)

        # MSE, MAE, MDAE
        test_mse = mean_squared_error(realisations.flatten(), mean_predictions.flatten())
        test_mae = mean_absolute_error(realisations.flatten(), mean_predictions.flatten())
        test_mdae = median_absolute_error(realisations.flatten(), mean_predictions.flatten())

        # CRPS_sum and Variogram_score
        test_crps_sum = np.mean([metrics.crps_sum(samples[i], realisations[i]) for i in range(realisations.shape[0])])
        test_variogram_score = np.mean([metrics.variogram_score(samples[i], realisations[i], weights="local", window_size=2) for i in range(realisations.shape[0])])

        # Store data in dataframe
        results.loc[results.shape[0]] = {
            **parameters,
            "trial_number" : trial_number,
            "trial_start" : trial_start,
            "trial_end" : time.time(),
            "validation_mse": val_mse,
            "validation_mae": val_mae,
            "validation_mdae": val_mdae ,
            "validation_crps_sum": val_crps_sum,
            "validation_variogram": val_variogram_score,
            "test_mse": test_mse,
            "test_mae": test_mae,
            "test_mdae": test_mdae ,
            "test_crps_sum": test_crps_sum,
            "test_variogram": test_variogram_score,
        }

        # Save results in csv file
        results.to_csv(f"results/test_exchange.csv", index=False)
            

if __name__ == "__main__":
    main()