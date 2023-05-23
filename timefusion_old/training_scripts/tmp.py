import time

# Time at start
_start_time = time.time()

print(f"Script started - Current runtime: {time.time() - _start_time}")

# Import necessary libraries
import random
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from timefusion import TimeFusion
from torch import nn
from diffusion import BatchLoader

print(f"Finished imports - Current runtime: {time.time() - _start_time}")

# Create dataset of two correlated, random walks on top of sine functions
# Set random seed to make result reproducible
np.random.seed(0)

# Random walk starts at 0,0
z = np.array([0,0])

# Covariance matrix for normal distribution
cov = [
    [1, 0.5],
    [0.5, 1]
]

# List to hold samples
samples = []

for i in range(60000):

    # Calculate underlying sine values
    y = 150*np.array([np.sin(i/40),np.sin((i/25))])
    #y = 1*np.array([np.sin(0.04*i),np.sin(0.04*i)])

    # Draw random samples from normal distribution
    z = np.random.multivariate_normal(z,cov)
    #z = np.array([0,0])

    # Store samples
    samples.append(y + z)

# Create pandas DataFrame out of data
data = pd.DataFrame(data=samples,columns=["sine1","sine2"])

# Remove 50% of samples to make into an irregular time-series
keep = 0.20 # What fraction of cells to keep
mask = [False]*int(2*len(data)*keep) + [True]*int(2*len(data)*(1-keep))
random.shuffle(mask)
mask = np.array(mask).reshape((len(data),2))
data = data.mask(mask)

# Remove all rows without any data
data = data.dropna(axis = 0, how = 'all')

data -= data.rolling(100,min_periods=30).mean()
data /= data.rolling(200,min_periods=60).std()

data = data[1000:]

print(f"Finished creating data - Current runtime: {time.time() - _start_time}")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device} - Current runtime: {time.time() - _start_time}")

# Make the data into BatchLoaders
train_loader = BatchLoader(
    data = data[:int(0.2*len(data))],
    batch_size = 64,
    context_length = 50,
    prediction_length = 30,
    diff_steps = 100,
    device = device
)

val_loader = BatchLoader(
    data = data[int(0.95*len(data)):],
    batch_size = 64,
    context_length = 50,
    prediction_length = 30,
    diff_steps = 100,
    device = device
)

print(f"BatchLoaders created - Current runtime: {time.time() - _start_time}")

predictor = TimeFusion(
    context_length = 50,
    prediction_length = 30,
    timeseries_shape = (2,3), 
    num_encoder_layers=3,
    d_model=128,
    nhead=2,
    dim_feedforward=128,
    diff_steps=100,
    device = device,
)

print("Number of trainable parameters:",sum(p.numel() for p in predictor.parameters()))

optimizer = torch.optim.Adam(params=predictor.parameters(),lr=3e-4)
lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=60)

print(f"Starting training - Current runtime: {time.time() - _start_time}")

predictor.train_network(
    train_loader = train_loader,
    epochs=20,
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

