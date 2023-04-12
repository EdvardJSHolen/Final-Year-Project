# This script processes the full dataset found at https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014#
# The main steps of the script are:
# 1. Remove data before and including 2012-01-01 00:00:00 as this part of the data is very incomplete
# 2. Resample the data to hourly frequency
# 3. Remove columns which are zero at first timestamp as these time-series are incomplete past 2012-01-01 00:00:00
# 4. Split out the data after 2014-10-01 00:00:00 as test data, giving 3 months of test data and 33 months of training data
# 5. Save the training and test data to separate csv files

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

# Remove columns which have a value of 0 at first timestamp
data = data[data.columns[data.iloc[0] != 0]]

# Split data into training and testing
train_data = data[data.index <= "2014-10-01"]
test_data = data[data.index > "2014-10-01"]

# Save train and test data to csv files
train_data.to_csv("train.csv")
test_data.to_csv("test.csv")
data.to_csv("full_data.csv")
