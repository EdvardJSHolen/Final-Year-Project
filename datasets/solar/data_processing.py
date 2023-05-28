# This script processes the full dataset found at https://www.nrel.gov/grid/assets/downloads/al-pv-2006.zip, https://www.nrel.gov/grid/solar-power-data.html
# The main steps of the script are:
# 1. Fetch actual data for all of 2006
# 2. Downsample data to 60 minute intervals
# 3. Split data into training, validation and testing sets with 80%, 10% and 10% of the data respectively
# 4. Save the datasets to csv files

# Import necesarry libraries
import pandas as pd
import os

# Locate all csv files for 60 minute data
directory = "al-pv-2006"
csv_files = [directory + "/" + file for file in os.listdir(directory) if file.startswith("Actual")]

# Read in the full data
data = pd.concat(map(lambda x: pd.read_csv(x, header=0, index_col=0), csv_files),axis=1)
data.index = pd.to_datetime(data.index)

# Downsample data to 60 minute intervals
data = data.resample("60min").mean()

# Split data into training, validation and testing sets
train_data = data.iloc[:int(0.8*len(data))]
val_data = data.iloc[int(0.8*len(data)):int(0.9*len(data))]
test_data = data.iloc[int(0.9*len(data)):]

# Save data to csv files
train_data.to_csv("train.csv")
val_data.to_csv("val.csv")
test_data.to_csv("test.csv")

