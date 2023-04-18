def main():   
    import time

    # Time at start
    _start_time = time.time()

    print(f"Script started - Current runtime: {time.time() - _start_time}")

    # Import necessary libraries
    import sys
    import random
    import torch
    import os
    import math
    from torch import nn
    from torch.utils.data import DataLoader
    import numpy as np
    import pandas as pd

    # Set path to fix relative imports
    sys.path.append("..")
    from data import TimeFusionDataset
    from timefusion import TimeFusion, EarlyStopper

    print(f"Finished package imports - Current runtime: {time.time() - _start_time}")

    # Import dataset
    train_data = pd.read_csv("../../datasets/electricity/train.csv").set_index("date")
    test_data = pd.read_csv("../../datasets/electricity/test.csv").set_index("date")
    train_data = train_data.iloc[:,:30]
    test_data = test_data.iloc[:,:30]
    train_data.index = pd.to_datetime(train_data.index)
    test_data.index = pd.to_datetime(test_data.index)

    # Standardize data
    means = train_data.mean()
    stds = train_data.std()
    train_data = (train_data-means)/stds
    test_data = (test_data-means)/stds

    # Randomly remove 30% of data to make irregular
    np.random.seed(0) # Set random seed to make result reproducible
    #remove = 0.30
    remove = 0

    # Training data 
    train_mask = np.full(train_data.size, False)
    train_mask[:int(train_data.size*remove)] = True
    np.random.shuffle(train_mask)
    train_data = train_data.mask(train_mask.reshape(train_data.shape))

    # Test data
    test_mask = np.full(test_data.size, False)
    test_mask[:int(test_data.size*remove)] = True
    np.random.shuffle(test_mask)
    test_data = test_data.mask(test_mask.reshape(test_data.shape))

    print(f"Finished importing data - Current runtime: {time.time() - _start_time}")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("cpu")
        #device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device} - Current runtime: {time.time() - _start_time}")

    # Define some common variables
    context_length = 48 
    prediction_length = 24

    encodings = [
        #lambda x: math.sin(2*math.pi*x.timestamp() / (3600*24)),
        lambda x: math.sin(2*math.pi*x.timestamp() / (3600*24*7)),
        lambda x: math.sin(2*math.pi*x.timestamp() / (3600*24*30)),
        #lambda x: math.sin(2*math.pi*x.timestamp() / (3600*24*90)),
        lambda x: math.sin(2*math.pi*x.timestamp() / (3600*24*365)),
        #lambda x: math.cos(2*math.pi*x.timestamp() / (3600*24)),
        lambda x: math.cos(2*math.pi*x.timestamp() / (3600*24*7)),
        lambda x: math.cos(2*math.pi*x.timestamp() / (3600*24*30)),
        #lambda x: math.cos(2*math.pi*x.timestamp() / (3600*24*90)),
        lambda x: math.cos(2*math.pi*x.timestamp() / (3600*24*365)),
    ]

    # Create each dataset
    train_dataset = TimeFusionDataset(
        data = train_data.iloc[:int(0.9*len(train_data))],
        context_length = context_length,
        prediction_length = prediction_length,
        timestamp_encodings = encodings
    )

    val_dataset = TimeFusionDataset(
        data = train_data.iloc[int(0.9*len(train_data)):],
        context_length = context_length,
        prediction_length = prediction_length,
        timestamp_encodings = encodings
    )

    train_loader = DataLoader(
        dataset = train_dataset,
        shuffle = True,
        num_workers = 4,
        batch_size = 128,
        #pin_memory=True,
        #pin_memory_device="cuda:0"
    )

    val_loader = DataLoader(
        dataset = val_dataset,
        shuffle = True,
        num_workers = 4,
        batch_size = 128,
        #pin_memory=True,
        #pin_memory_device="cuda:0"
    )

    print(f"DataLoaders created - Current runtime: {time.time() - _start_time}")

    predictor = TimeFusion(
        context_length = context_length,
        prediction_length = prediction_length,
        timeseries_shape = (len(train_dataset.time_series),train_dataset.time_series[0].shape[1]), 
        num_encoder_layers=3,
        d_model=512,
        nhead=32,
        dim_feedforward=1024,
        diff_steps=100,
        device = device,
    )

    print("Number of trainable parameters:",sum(p.numel() for p in predictor.parameters()))

    print([p.numel() for p in predictor.parameters()])

    optimizer = torch.optim.Adam(params=predictor.parameters(),lr=2e-4)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=20)

    print(f"Starting training - Current runtime: {time.time() - _start_time}")

    predictor.train_network(
        train_loader = train_loader,
        epochs=20,
        val_loader = val_loader,
        val_metrics= {
            "Val MAE": nn.L1Loss(),
        },
        optimizer = optimizer,
        lr_scheduler= lr_scheduler,
        early_stopper=EarlyStopper(patience=10)
    )

    print(f"Finished training - Current runtime: {time.time() - _start_time}")

    if not os.path.exists("weights"):
        os.makedirs("weights")

    torch.save(predictor.state_dict(), "weights/" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime()))

    print(f"Saved weights - Current runtime: {time.time() - _start_time}")

    print(f"Script ended - Total runtime: {time.time() - _start_time}")

if __name__ == "__main__":
    main()