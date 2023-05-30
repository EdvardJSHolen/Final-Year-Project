# This script processes the full dataset found at https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014#
# The main steps of the script are:
# 1. Remove data before and including 2012-01-01 00:00:00 as this part of the data is very incomplete
# 2. Remove last month of data as this is incomplete, especially around christmas
# 3. Resample the data to hourly frequency
# 4. Remove columns which are zero at first timestamp as these time-series are incomplete past 2012-01-01 00:00:00
# 5. Split data into training, validation and testing sets with 80%, 10% and 10% of the data respectively
# 6. Save the datasets to csv files

# Import necesarry libraries
import pandas as pd
from collections import defaultdict

# Read in the full data
data = pd.read_csv(
    filepath_or_buffer = 'original.csv',
    names = ["date"] + list(range(370)),
    sep=";",
    header=0,
    index_col="date",
    dtype=defaultdict(lambda: "float", {"date": "string"}),
    decimal=","
) 
data.index = pd.to_datetime(data.index)

# Resample data to hourly frequency
data = data[data.index > "2012-01-01"].resample("1h").ffill().dropna()
data = data[data.index < "2014-12-01"]

# Remove columns which have a value of 0 at first timestamp
data = data[data.columns[data.iloc[0] != 0]]

# Split data into training, validation and testing sets
train_data = data.iloc[:int(0.8*len(data))]
val_data = data.iloc[int(0.8*len(data)):int(0.9*len(data))]
test_data = data.iloc[int(0.9*len(data)):]

# Save train and test data to csv files
train_data.to_csv("train.csv")
val_data.to_csv("val.csv")
test_data.to_csv("test.csv")
