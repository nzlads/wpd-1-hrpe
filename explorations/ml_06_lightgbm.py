#%% packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from hrpe.data.load import load_hh_data, load_maxmin_data, load_weather_data
from hrpe.features.weather import interpolate_weather
from hrpe.features.deltas import calculate_deltas
from hrpe.features.time import make_datetime_features
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

train = data.query("type == 'pre_august'")
test = data.query("type == 'august'")

# %% Fit a LightGBM Model I guess, without lags to start
dmax_covariates = [
    "value_mean",
    "period",
    "day_of_week",
    "is_weekday",
    "temperature",
    "solar_irradiance",
]
dmin_covariates = [
    "value_mean",
    "period",
    "day_of_week",
    "is_weekday",
    "temperature",
    "solar_irradiance",
]

dmax_lgbm = LightGBMModel(lags=None, lags_future_covariates=[0] * len(dmax_covariates))
dmin_lgbm = LightGBMModel(lags=None, lags_future_covariates=[0] * len(dmin_covariates))


dmax_ts = TimeSeries.from_dataframe(train, time_col="time", value_cols="delta_max")
dmin_ts = TimeSeries.from_dataframe(train, time_col="time", value_cols="delta_min")

dmax_cov = TimeSeries.from_dataframe(train, time_col="time", value_cols=dmax_covariates)
dmin_cov = TimeSeries.from_dataframe(train, time_col="time", value_cols=dmin_covariates)

dmax_lgbm.fit(dmax_ts, future_covariates=dmax_cov)
dmin_lgbm.fit(dmin_ts, future_covariates=dmin_cov)

# %% Predict, so need to make test covariates and predictions etc
dmax_cov_test = TimeSeries.from_dataframe(
    test, time_col="time", value_cols=dmax_covariates
)
dmin_cov_test = TimeSeries.from_dataframe(
    test, time_col="time", value_cols=dmin_covariates
)

dmax_preds = dmax_lgbm.predict(
    n=len(test), future_covariates=dmax_cov_test
).pd_dataframe()
dmin_preds = dmin_lgbm.predict(
    n=len(test), future_covariates=dmin_cov_test
).pd_dataframe()

preds = pd.DataFrame.join(dmax_preds, dmin_preds).join(
    test[["time", "value_mean"]].set_index("time")
)
preds["value_max"] = preds["value_mean"] + preds["delta_max"]
preds["value_min"] = preds["value_mean"] - preds["delta_min"]

truths = test[["time", "value_max", "value_min", "value_mean"]]
print(score_model(preds.reset_index(), truths.reset_index(drop=True)))
# With average is 0.4522, way better than ETS basic model
# With station 1 is 0.447, also way better than basic (no other covariate) models

# %% Attempt using custom class
# Restate for convenience of reading
dmax_covariates = [
    "value_mean",
    "period",
    "day_of_week",
    "is_weekday",
    "temperature",
    "solar_irradiance",
]
dmin_covariates = [
    "value_mean",
    "period",
    "day_of_week",
    "is_weekday",
    "temperature",
    "solar_irradiance",
]


lgbm = DartsLGBMModel(
    dmax_covariates=dmax_covariates,
    dmin_covariates=dmin_covariates,
    dmax_lags={"lags": None, "lags_future_covariates": [0] * len(dmax_covariates)},
    dmin_lags={"lags": None, "lags_future_covariates": [0] * len(dmin_covariates)},
)

lgbm.fit(train)
custom_preds = lgbm.predict(
    forecast=test[["time", "value_mean"]], future_covariates_df=test
)

truths = test[["time", "value_max", "value_min", "value_mean"]].reset_index(drop=True)
score_model(custom_preds, truths)
# Works!
# %% Attempt custom model with lags
lgbm_lags = DartsLGBMModel(
    dmax_covariates=dmax_covariates,
    dmin_covariates=dmin_covariates,
    dmax_lags={"lags": 48, "lags_future_covariates": [0] * len(dmax_covariates)},
    dmin_lags={"lags": 48, "lags_future_covariates": [0] * len(dmin_covariates)},
)
lgbm_lags.fit(train)
lags_preds = lgbm_lags.predict(
    forecast=test[["time", "value_mean"]], future_covariates_df=data
)

truths = test[["time", "value_max", "value_min", "value_mean"]].reset_index(drop=True)
score_model(lags_preds, truths)
# Also works! This particular one happens to be a lil worse than the previous
