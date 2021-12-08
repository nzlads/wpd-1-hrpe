# %% packages
import pandas as pd

from hrpe.data.load import load_hh_data, load_maxmin_data, load_weather_data
from hrpe.features.weather import interpolate_weather
from hrpe.features.deltas import calculate_deltas
from hrpe.features.time import make_datetime_features
from hrpe.models.selection import MonthSeriesSplit
from hrpe.models.eval import score_model

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
msplit = MonthSeriesSplit(n_splits=5, min_train_months=12)

# %% Define cross validate function
def fit_and_score_delta(train, test, model, target, verbose=True):
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
    return [test_start, mse_score]


n_jobs = cpu_count() - 2
print(n_jobs)

best_covariates = ["value_mean", "period", "temperature", "solar_irradiance"]
dmax_model = DeltaLGBMModel(
    target="max",
    covariates=best_covariates,
    lags={"lags": None, "lags_future_covariates": [0] * len(best_covariates)},
)

parallel = Parallel(n_jobs=n_jobs, verbose=True)
scores = parallel(
    delayed(fit_and_score_delta)(train, test, dmax_model, "delta_max")
    for train, test in msplit.split(data)
)
# %% Again but for full combo for min
all_covariates = [
    "value_mean",
    "period",
    "day_of_week",
    # "is_weekday",
    "month_of_year",
    "temperature",
    "solar_irradiance",
    "windspeed_north",
    "windspeed_east",
    "pressure",
    "spec_humidity",
]


def fit_score_delta_model(train, test, covs, data, target):
    delta_target = f"delta_{target}"
    model = DeltaLGBMModel(
        target=target,
        covariates=covs,
        lags={"lags": None, "lags_future_covariates": [0] * len(covs)},
    )
    test_start = test.time.min()
    model.fit(train)
    preds = model.predict(len(test), future_covariates_df=data).reset_index()
    truths = test[["time", delta_target]]
    mse_score = mse(
        actual_series=TimeSeries.from_dataframe(truths, "time", delta_target),
        pred_series=TimeSeries.from_dataframe(preds, "time", delta_target),
    )
    cov_list = ", ".join(covs)
    return [cov_list, test_start, mse_score]


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def gen_combos(data, split, covs):
    for cov_set in powerset(covs):
        for train, test in split.split(data):
            yield train, test, list(cov_set)


n_jobs = cpu_count() - 1
parallel = Parallel(n_jobs=n_jobs, verbose=True)
min_scores = parallel(
    delayed(fit_score_delta_model)(train, test, covs, data, "min")
    for train, test, covs in gen_combos(data, msplit, all_covariates)
)

# %%
