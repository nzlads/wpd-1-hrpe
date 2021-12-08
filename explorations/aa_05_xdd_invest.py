#%% packages

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from hrpe.data.load import load_hh_data, load_maxmin_data, load_weather_data
from hrpe.features.weather import add_weather, interpolate_weather
from hrpe.features.deltas import calculate_deltas
from hrpe.features.time import make_datetime_features
from hrpe.models.selection import MonthSeriesSplit
from hrpe.models.eval import score_model
from hrpe.models.darts import DartsLGBMModel, DeltaLGBMModel

from itertools import chain, combinations


# %% Load data
SUBSTATION = "staplegrove"

hh_data = load_hh_data(SUBSTATION)
maxmin_data = load_maxmin_data(SUBSTATION)
weather_data = load_weather_data(SUBSTATION)
weather_data = interpolate_weather(weather_data)
weather_data = weather_data.query("station == '1'")
demand_data = pd.merge(maxmin_data, hh_data, on="time").rename(
    columns={"value": "value_mean"}
)
demand_data = calculate_deltas(demand_data)
demand_data = make_datetime_features(demand_data)
data = add_weather(demand_data, weather_data)


# %%
xdata = xdd(data, 15, 5)


# %%
plt.figure(figsize=(10, 10))
plt.plot(xdata.time, 10 * xdata.delta_max)
plt.plot(xdata.time, xdata.temperature)
plt.show()


plt.plot(xdata.time, 10 * xdata.delta_min)
plt.plot(xdata.time, xdata.temperature)
plt.show()


plt.scatter(xdata.hdd, xdata.delta_min)
