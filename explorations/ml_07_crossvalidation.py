#%% packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from hrpe.data.load import load_hh_data, load_maxmin_data, load_weather_data
from hrpe.features.weather import interpolate_weather
from hrpe.features.deltas import calculate_deltas
from hrpe.features.time import make_datetime_features
from hrpe.models.selection import MonthSeriesSplit
from hrpe.models.eval import score_model

from hrpe.models.darts import DartsLGBMModel

# Fuck it just try lightgbm?
from darts import TimeSeries
from darts.models import LightGBMModel

#%% Load data
SUBSTATION = "staplegrove"

hh_data = load_hh_data(SUBSTATION)
maxmin_data = load_maxmin_data(SUBSTATION)
weather_data = load_weather_data(SUBSTATION)
weather_data = interpolate_weather(weather_data)
# Average the weather data
# weather_data = weather_data.groupby("time").agg("mean")
# Alternatively, only take station 1
weather_data = weather_data.query("station == '1'")

#%% Merge data
demand_data = pd.merge(maxmin_data, hh_data, on="time").rename(
    columns={"value": "value_mean"}
)
demand_data = calculate_deltas(demand_data)
demand_data = make_datetime_features(demand_data)
data = pd.merge(demand_data, weather_data, on="time")

# %% create test-train generator


def month_series_split(data: pd.DataFrame, n_splits=5, min_train_months=12):
    """
    Generate test and train splits for cross-validation by month.
    For each split, the test set is 1 month of data.
    Splits are moved backward one month at a time.


    Parameters
    ----------
    data: pd.DataFrame
        Data to be split

    n_splits: int, defualt=5
        Number of splits

    min_train_months: int, default=12
        Minimum number of months in training set



    """
    assert "time" in data.columns, "data must contain 'time' column."

    # Make sure there's enough months in data to do all splits
