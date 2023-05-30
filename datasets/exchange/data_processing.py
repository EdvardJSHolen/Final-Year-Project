# This script processes the full dataset found at https://github.com/laiguokun/multivariate-time-series-data/blob/master/exchange_rate/exchange_rate.txt.gz
# The main steps of the script are:
# 1. Split data into training, validation and testing sets with 80%, 10% and 10% of the data respectively
# 2. Save the datasets to csv files

# Import necesarry libraries
import pandas as pd

# Read in the full data
data = pd.read_csv(
    filepath_or_buffer = "exchange_rate.txt",
    names = list(range(8)),
    sep=",",
    header=0,
    decimal="."
)

# Split data into training, validation and testing sets
train_data = data.iloc[:int(0.8*len(data))]
val_data = data.iloc[int(0.8*len(data)):int(0.9*len(data))]
test_data = data.iloc[int(0.9*len(data)):]

# Save data to csv files
train_data.to_csv("train.csv")
val_data.to_csv("val.csv")
test_data.to_csv("test.csv")
