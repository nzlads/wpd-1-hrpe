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
msplit = MonthSeriesSplit(n_splits=8, min_train_months=12)

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
parallel = Parallel(n_jobs=n_jobs, verbose=3)
min_scores = parallel(
    delayed(fit_score_delta_model)(train, test, covs, data, "min")
    for train, test, covs in gen_combos(data, msplit, all_covariates)
)
min_scores = (
    pd.DataFrame(min_scores, columns=["covariates", "test_start", "mse"])
    .groupby("covariates")
    .agg({"mse": "mean"})
    .reset_index()
    .sort_values('mse')
)
min_scores.to_csv("data/processed/min_cv_scores.csv", index=False)

# %% Repeat for max
max_scores = parallel(
    delayed(fit_score_delta_model)(train, test, covs, data, "max")
    for train, test, covs in gen_combos(data, msplit, all_covariates)
)
max_scores = (
    pd.DataFrame(max_scores, columns=["covariates", "test_start", "mse"])
    .groupby("covariates")
    .agg({"mse": "mean"})
    .reset_index()
    .sort_values('mse')
)
max_scores.to_csv("data/processed/max_cv_scores.csv", index=False)

# %% Use best covariates for each of max and min
best_max_covariates = [
    "value_mean", "period", "month_of_year", "temperature"
]
best_min_covariates = [
    "value_mean", "period", "month_of_year", "temperature", "solar_irradiance", "windspeed_east", "spec_humidity"
]

lgbm = DartsLGBMModel(
    dmax_covariates=best_max_covariates,
    dmin_covariates=best_min_covariates,
    dmax_lags={"lags": None, "lags_future_covariates": [0] * len(best_max_covariates)},
    dmin_lags={"lags": None, "lags_future_covariates": [0] * len(best_min_covariates)},
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