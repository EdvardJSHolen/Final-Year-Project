import time

# Time at start
_start_time = time.time()

print(f"Script started - Current runtime: {time.time() - _start_time}")

# Set path to fix relative imports
import sys
sys.path.append("..")

# Import necessary libraries
import random
import torch
import os
from torch import nn
import numpy as np
import pandas as pd
from diffusion import BatchLoader
from timefusion import TimeFusion

print(f"Finished package imports - Current runtime: {time.time() - _start_time}")

# Import dataset
train_data = pd.read_csv("../../datasets/electricity/train.csv").set_index("date")
test_data = pd.read_csv("../../datasets/electricity/test.csv").set_index("date")
train_data.index = pd.to_datetime(train_data.index)
test_data.index = pd.to_datetime(test_data.index)

# Randomly remove 30% of data to make irregular
np.random.seed(0) # Set random seed to make result reproducible
remove = 0.30

# Training data 
train_mask = np.full(train_data.size, False)
train_mask[-int(train_data.size*remove):] = True
np.random.shuffle(train_mask)
train_data = train_data.mask(train_mask.reshape(train_data.shape))

# Test data
test_mask = np.full(test_data.size, False)
test_mask[-int(test_data.size*remove):] = True
np.random.shuffle(test_mask)
test_data = test_data.mask(test_mask.reshape(test_data.shape))

print(f"Finished importing data - Current runtime: {time.time() - _start_time}")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device} - Current runtime: {time.time() - _start_time}")

# Make the data into BatchLoaders
train_loader = BatchLoader(
    data = train_data[:int(0.9*len(train_data))],
    batch_size = 128,
    context_length = 50,
    prediction_length = 30,
    diff_steps = 100,
    device = device,
    lazy_init= True
)

val_loader = BatchLoader(
    data = train_data[int(0.9*len(train_data)):],
    batch_size = 128,
    context_length = 50,
    prediction_length = 30,
    diff_steps = 100,
    device = device,
    lazy_init= True
)

test_loader = BatchLoader(
    data = test_data,
    batch_size = 128,
    context_length = 50,
    prediction_length = 30,
    diff_steps = 100,
    device = device,
    lazy_init= True
)

print(f"BatchLoaders created - Current runtime: {time.time() - _start_time}")

predictor = TimeFusion(
    context_length = 50,
    prediction_length = 30,
    timeseries_shape = (319,10), 
    num_encoder_layers=3,
    d_model=2048,
    nhead=8,
    dim_feedforward=2048,
    diff_steps=100,
    device = device,
)

print("Number of trainable parameters:",sum(p.numel() for p in predictor.parameters()))

optimizer = torch.optim.Adam(params=predictor.parameters(),lr=1e-4)
lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=60)

print(f"Starting training - Current runtime: {time.time() - _start_time}")

predictor.train_network(
    train_loader = train_loader,
    epochs=2,
    val_loader = val_loader,
    val_metrics= {
        "Val MAE": nn.L1Loss(),
        "Val MSE": nn.MSELoss()
    },
    optimizer = optimizer,
    lr_scheduler= lr_scheduler
)

print(f"Finished training - Current runtime: {time.time() - _start_time}")

if not os.path.exists("weights"):
   os.makedirs("weights")

torch.save(predictor, "weights/" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime()))

print(f"Saved weights - Current runtime: {time.time() - _start_time}")

print(f"Script ended - Total runtime: {time.time() - _start_time}")

