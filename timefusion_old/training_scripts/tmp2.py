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
    import numpy as np
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader

    # Set path to fix relative imports
    sys.path.append("..")
    from data import TimeFusionDataset

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
    remove = 0.30
    #remove = 0

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
        #device = torch.device("cpu")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device} - Current runtime: {time.time() - _start_time}")


    a = TimeFusionDataset(
        data = train_data[:int(0.9*len(train_data))],
        context_length = 48,
        prediction_length = 24,
    )
    print(f"Finished creating dataset - Current runtime: {time.time() - _start_time}")

    # create a DataLoader that uses the CustomDataset with 2 worker processes
    batch_size = 128
    num_workers = 4
    custom_dataloader = DataLoader(a, 
        batch_size=batch_size, 
        num_workers=num_workers,
        shuffle=True,
        #pin_memory= True if device == torch.device("cuda:0") else False,
        #pin_memory_device="cuda:0"
    )

    # iterate over the dataloader
    for batch in custom_dataloader:
        print(batch.shape)
        #print(batch.device)
        #batch.to(torch.device("cuda:0"))

    # iterate over the dataloader
    for batch in custom_dataloader:
        print(batch.shape)
        #print(batch.device)
        #batch.to(torch.device("cuda:0"))

    print(f"Script finished - Current runtime: {time.time() - _start_time}")


if __name__ == '__main__':
    main()
