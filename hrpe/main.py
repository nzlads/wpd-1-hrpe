import os


import pandas as pd

from hrpe.data.load import load_hh_data, load_maxmin_data, load_weather_data
from hrpe.features.weather import interpolate_weather
from hrpe.features.deltas import calculate_deltas
from hrpe.features.time import make_datetime_features
from hrpe.models.darts import DartsLGBMModel
from hrpe.submit.create_submission import create_submissions_file


def main():

    print(os.getcwd())

    # Set Vars
    substation = "staplegrove"

    # Load data using load.py function for staplegrove
    # Load data function

    hh_data = load_hh_data(substation=substation)
    maxmin_data = load_maxmin_data(substation=substation)
    weather_data = load_weather_data(substation=substation)

    # Build features / differences
    weather_data = interpolate_weather(weather_data)
    weather_data = weather_data.query("station == '1'")

    # Merge data
    demand_data = pd.merge(maxmin_data, hh_data, on="time", how="outer").rename(
        columns={"value": "value_mean"}
    )
    demand_data = calculate_deltas(demand_data)
    demand_data = make_datetime_features(demand_data)
    data = pd.merge(demand_data, weather_data, on="time")
    train = data.query("type != 'september'")
    test = data.query("type == 'september'")

    best_max_covariates = ["value_mean", "period", "month_of_year", "temperature"]
    best_min_covariates = [
        "value_mean",
        "period",
        "month_of_year",
        "temperature",
        "solar_irradiance",
        "windspeed_east",
        "spec_humidity",
    ]

    lgbm = DartsLGBMModel(
        dmax_covariates=best_max_covariates,
        dmin_covariates=best_min_covariates,
        dmax_lags={
            "lags": None,
            "lags_future_covariates": [0] * len(best_max_covariates),
        },
        dmin_lags={
            "lags": None,
            "lags_future_covariates": [0] * len(best_min_covariates),
        },
    )
    lgbm.fit(train)
    preds = lgbm.predict(
        forecast=test[["time", "value_mean"]], future_covariates_df=data
    )
    # print(preds)
    create_submissions_file(preds, "september", "ml")

    return None


# Model
# Naive fit
# naive predict


# Scoring
# RMSE score


# Debugging plots


##
# standards:
# minmax = truth
# hh = halfhour


if __name__ == "__main__":
    # execute only if run as the entry point into the program
    main()
