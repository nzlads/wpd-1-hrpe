import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from hrpe.data.load import load_minute_data
from hrpe.features.time import make_datetime_features
from hrpe.models.eval import score_model

# load data
data = load_minute_data("staplegrove")
data = make_datetime_features(data)

# First exploration: how many periods have bad data points?
# badpp = data.groupby("period_time").agg({'quality': lambda x: x.isin(["Bad", "Bad Ip"]).sum()})
# print(f"There are {sum(badpp.quality > 0)} periods with bad data out of {len(badpp)}")
# print(f"There are {sum(badpp.quality == 30)} periods that are entirely bad")
# # 94 of 30672 periods have some bad data, is 0.3%
# # 51 periods have entirely bad data data (all 30 mins are bad)
# # 0.3% is not much to drop so just drop it
# badpp["is_bad"] = badpp["quality"] > 0
# data = data.merge(badpp[["is_bad"]], on="period_time")
# data = data[~data["is_bad"]]

# Calculate the half-hourly features data from the minute data
hh_data = data.groupby("period_time").agg({"value": ["max", "min", "mean"]})
hh_data.columns = ['_'.join(col) for col in hh_data.columns.to_flat_index()]
hh_data["time"] = hh_data.index
hh_data = make_datetime_features(hh_data).reset_index(drop=True)

hh_data["delta_max"] = hh_data["value_max"] - hh_data["value_mean"]
hh_data["delta_min"] = hh_data["value_mean"] - hh_data["value_min"]
hh_data["is_weekday"] = hh_data["is_weekday"].astype(int)

# Define base class for model
