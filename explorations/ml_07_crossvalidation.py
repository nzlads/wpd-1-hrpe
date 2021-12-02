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

from hrpe.models.darts import DartsLGBMModel, DeltaLGBMModel

from itertools import chain, combinations

# Fuck it just try lightgbm?
from darts import TimeSeries
from darts.models import LightGBMModel
from darts.metrics import mse

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
# These are default params but stating for transparency
msplit = MonthSeriesSplit(n_splits=5, min_train_months=12)

dmax_covariates = [
    "value_mean",
    "period",
    "day_of_week",
    "is_weekday",
    "temperature",
    "solar_irradiance",
]

dmax_model = DeltaLGBMModel(
    target="max",
    covariates=dmax_covariates,
    lags={"lags": None, "lags_future_covariates": [0] * len(dmax_covariates)},
)

# %%

all_scores = []
for train, test in msplit.split(data):
    test_start = test.time.min()
    print(f"Fitting for {test_start}")
    dmax_model.fit(train)
    preds = dmax_model.predict(len(test), future_covariates_df=data).reset_index()
    truths = test[["time", "delta_max"]]
    mse_score = mse(
        actual_series=TimeSeries.from_dataframe(truths, "time", "delta_max"),
        pred_series=TimeSeries.from_dataframe(preds, "time", "delta_max"),
    )
    all_scores.append([test_start, mse_score])
scores = pd.DataFrame(all_scores)
scores.columns = ["test_start", "mse"]
print(scores)

# %% Compare with one random covariate
dmax_covariates = ["period"]
dmax_model = DeltaLGBMModel(
    target="max",
    covariates=dmax_covariates,
    lags={"lags": None, "lags_future_covariates": [0] * len(dmax_covariates)},
)

all_scores = []
for train, test in msplit.split(data):
    test_start = test.time.min()
    print(f"Fitting for {test_start}")
    dmax_model.fit(train)
    preds = dmax_model.predict(len(test), future_covariates_df=data).reset_index()
    truths = test[["time", "delta_max"]]
    mse_score = mse(
        actual_series=TimeSeries.from_dataframe(truths, "time", "delta_max"),
        pred_series=TimeSeries.from_dataframe(preds, "time", "delta_max"),
    )
    all_scores.append([test_start, mse_score])
scores = pd.DataFrame(all_scores)
scores.columns = ["test_start", "mse"]
print(scores)

# %% Now functionalise some shit
def cross_validate(model, split, data, target, verbose=False):
    scores = []
    for train, test in split.split(data):
        test_start = test.time.min()
        if verbose:
            print(f"Fitting for data up to {test_start}")
        model.fit(train)
        preds = model.predict(len(test), future_covariates_df=data).reset_index()
        truths = test[["time", target]]
        mse_score = mse(
            actual_series=TimeSeries.from_dataframe(truths, "time", target),
            pred_series=TimeSeries.from_dataframe(preds, "time", target),
        )
        scores.append([test_start, mse_score])
    scores = pd.DataFrame(scores, columns=["test_start", "mse"])
    return scores


dmax_covariates = ["value_mean"]
dmax_model = DeltaLGBMModel(
    target="max",
    covariates=dmax_covariates,
    lags={"lags": None, "lags_future_covariates": [0] * len(dmax_covariates)},
)
cv_scores = cross_validate(dmax_model, msplit, data, "delta_max", verbose=True)
print(cv_scores)
# %% Now test various combinations
# WARNING: This takes 63 x 4 mins to run in its current form
# TODO: Parallelise me!
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


dmax_covariates = [
    "value_mean",
    "period",
    "day_of_week",
    "is_weekday",
    "temperature",
    "solar_irradiance",
]

all_scores = []
for cov_list in powerset(dmax_covariates):
    cov_list = list(cov_list)
    dmax_model = DeltaLGBMModel(
        target="max",
        covariates=cov_list,
        lags={"lags": None, "lags_future_covariates": [0] * len(cov_list)},
    )
    cv_scores = cross_validate(dmax_model, msplit, data, "delta_max", verbose=True)
    cv_scores["covariates"] = ", ".join(cov_list)
    all_scores.append(cv_scores)
all_scores = pd.concat(all_scores)
print(all_scores)
# %%
all_scores.groupby("covariates").agg({"mse": "mean"}).sort_values("mse")
# The covariates with the lowest avg mse across 5 splits is
# value_mean, period, temperature, solar_irradiance

# Now need to repeat for min, but I suspect that would be the same outcome, so just build a model with it:

# %% Best basic LGBM Model I think
best_covariates = ["value_mean", "period", "temperature", "solar_irradiance"]

lgbm = DartsLGBMModel(
    dmax_covariates=best_covariates,
    dmin_covariates=best_covariates,
    dmax_lags={"lags": None, "lags_future_covariates": [0] * len(best_covariates)},
    dmin_lags={"lags": None, "lags_future_covariates": [0] * len(best_covariates)},
)

skill_scores = []
for train, test in msplit.split(data):
    test_start = test.time.min()
    lgbm.fit(train)
    custom_preds = lgbm.predict(
        forecast=test[["time", "value_mean"]], future_covariates_df=test
    )
    truths = test[["time", "value_max", "value_min", "value_mean"]].reset_index(
        drop=True
    )
    skill_score = score_model(custom_preds, truths)
    skill_scores.append([test_start, skill_score])
skill_scores = pd.DataFrame(skill_scores, columns=["test_start", "skill_score"])
skill_scores
# August skill score 0.444 is better than the full covariate set
