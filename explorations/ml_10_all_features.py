# %% packages
import numpy as np
import pandas as pd

from hrpe.data.load import load_hh_data, load_maxmin_data, load_weather_data
from hrpe.features.weather import interpolate_weather, use_all_stations
from hrpe.features.deltas import calculate_deltas
from hrpe.features.time import make_datetime_features
from hrpe.models.selection import MonthSeriesSplit
from hrpe.models.eval import score_model
from hrpe.features.xdd import xdd

from hrpe.models.darts import DartsLGBMModel, DeltaLGBMModel

# Fuck it just try lightgbm?
from darts import TimeSeries
from darts.metrics import mse

from joblib import Parallel, delayed, cpu_count
from itertools import chain, combinations

#%% Load data
SUBSTATION = "staplegrove"

hh_data = load_hh_data(SUBSTATION)
maxmin_data = load_maxmin_data(SUBSTATION)
weather_data = load_weather_data(SUBSTATION)
weather_data = interpolate_weather(weather_data)
# weather_data = xdd(weather_data)
# Add all weather features
weather_data = use_all_stations(weather_data)

demand_data = pd.merge(maxmin_data, hh_data, on="time").rename(
    columns={"value": "value_mean"}
)
demand_data = calculate_deltas(demand_data)
demand_data = make_datetime_features(demand_data)
data = pd.merge(demand_data, weather_data, on="time")
msplit = MonthSeriesSplit(n_splits=10, min_train_months=12)

# %% CV on all covariates
all_covariates = list(
    set(data.columns)
    - {
        "time",
        "value_max",
        "value_min",
        "type",
        "delta_max",
        "delta_min",
        "range",
        "period_time",
        "is_weekday",
    }
)


def fit_and_score(train, test, model, data):
    test_start = test.time.min()
    model.fit(train)
    preds = model.predict(test[["time", "value_mean"]], future_covariates_df=data)
    truths = test[["time", "value_mean", "value_min", "value_max"]].reset_index(
        drop=True
    )
    skill_score = score_model(preds, truths)
    return [test_start, skill_score]


lags = np.array([47, 48, 49, 48 * 7 - 1, 48 * 7, 48 * 7 + 1]) * -1
lags_future_covariates = np.concatenate([np.arange(1, 49), np.array([96, 48 * 7])]) * -1

n_jobs = cpu_count() - 1
model = DartsLGBMModel(
    dmax_covariates=all_covariates,
    dmin_covariates=all_covariates,
    dmax_lags={
        "lags": lags.tolist(),
        "lags_future_covariates": lags_future_covariates.tolist(),
    },
    dmin_lags={
        "lags": lags.tolist(),
        "lags_future_covariates": lags_future_covariates.tolist(),
    },
    dmax_kwargs={"n_estimators": 300},
    dmin_kwargs={"n_estimators": 300},
)
parallel = Parallel(n_jobs=n_jobs, verbose=9)
# %%
scores = parallel(
    delayed(fit_and_score)(train, test, model, data)
    for train, test in msplit.split(data)
)
pd.DataFrame(scores, columns=["test_start", "skill_score"])

# %%
