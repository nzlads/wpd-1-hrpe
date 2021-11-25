import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from hrpe.data.load import load_minute_data, load_hh_data, load_maxmin_data
from hrpe.features.time import make_datetime_features
from hrpe.models.eval import score_model
from hrpe.models.periodic import SnaivePeriodModel

# load data
data = load_minute_data("staplegrove")
data = make_datetime_features(data)

# Calculate the half-hourly features data from the minute data
hh_data = data.groupby("period_time").agg({"value": ["max", "min", "mean"]})
hh_data.columns = ["_".join(col) for col in hh_data.columns.to_flat_index()]
hh_data["time"] = hh_data.index
hh_data = make_datetime_features(hh_data).reset_index(drop=True)

hh_data["delta_max"] = hh_data["value_max"] - hh_data["value_mean"]
hh_data["delta_min"] = hh_data["value_mean"] - hh_data["value_min"]
hh_data["is_weekday"] = hh_data["is_weekday"].astype(int)

# Define base class for model

mod = SnaivePeriodModel(seasonalities={"years": 1})
mod.fit(hh_data)


truths = load_maxmin_data("staplegrove", time_start="2021-07-01", time_end="2021-08-01")
forecast = load_hh_data("staplegrove", time_start="2021-07-01", time_end="2021-08-01")
forecast = forecast[["time", "value"]]
forecast.columns = ["time", "value_mean"]

truths = truths.merge(forecast, on="time")

preds = mod.predict(forecast)
score_model(preds, truths)
# 0.867
