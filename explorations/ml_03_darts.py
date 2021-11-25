import pandas as pd

from hrpe.data.load import (
    filter_data_by_time,
    load_minute_data,
    load_hh_data,
    load_maxmin_data,
)
from hrpe.features.deltas import calculate_deltas
from hrpe.features.transform import minute_data_to_hh_data
from hrpe.models.eval import score_model
from hrpe.models.ets import ETS
from hrpe.submit.create_submission import create_submissions_file

from darts import TimeSeries
from darts.utils.statistics import (
    check_seasonality,
    plot_acf,
    plot_residuals_analysis,
    plot_hist,
)
from darts.models import NaiveSeasonal, AutoARIMA, ExponentialSmoothing, Theta
from darts.metrics import mape, mse

data = load_minute_data("staplegrove")
hh_data = minute_data_to_hh_data(data)
hh_data = calculate_deltas(hh_data)

TRAIN_DATE = "2021-07-01"
hh_train = filter_data_by_time(hh_data, time_start=None, time_end=TRAIN_DATE)
hh_valid = filter_data_by_time(
    hh_data, time_start=TRAIN_DATE, time_end=None
).reset_index(drop=True)

# Use darts class
dmax_ts = TimeSeries.from_dataframe(
    hh_train, time_col="time", value_cols="delta_max", freq="30min"
)
dmin_ts = TimeSeries.from_dataframe(
    hh_train, time_col="time", value_cols="delta_min", freq="30min"
)

valid_dmax_ts = TimeSeries.from_dataframe(
    hh_valid, time_col="time", value_cols="delta_max", freq="30min"
)
valid_dmin_ts = TimeSeries.from_dataframe(
    hh_valid, time_col="time", value_cols="delta_min", freq="30min"
)


def eval_model(model, train_ts, valid_ts):
    model.fit(train_ts)
    forecast = model.predict(len(valid_ts))
    print("model {} obtains MSE: {:.2f}".format(model, mse(valid_ts, forecast)))


eval_model(ExponentialSmoothing(seasonal_periods=48), dmax_ts, valid_dmax_ts)
# eval_model(AutoARIMA(), dmax_ts, valid_dmax_ts)
eval_model(Theta(), dmax_ts, valid_dmax_ts)
eval_model(NaiveSeasonal(48), dmax_ts, valid_dmax_ts)

eval_model(ExponentialSmoothing(seasonal_periods=48), dmin_ts, valid_dmin_ts)
# eval_model(AutoARIMA(), dmin_ts, valid_dmin_ts)
eval_model(Theta(), dmin_ts, valid_dmin_ts)
eval_model(NaiveSeasonal(48), dmin_ts, valid_dmin_ts)

# Try create and calculate result for ETS model outputs
def fit_and_predict(model, train, n):
    model.fit(train)
    return model.predict(n).pd_dataframe()


preds = (
    fit_and_predict(ExponentialSmoothing(seasonal_periods=48), dmax_ts, 31 * 48)
    .join(fit_and_predict(ExponentialSmoothing(seasonal_periods=48), dmin_ts, 31 * 48))
    .join(hh_valid.set_index("time")[["value_mean"]])
    .reset_index()
)

preds["value_max"] = preds["value_mean"] + preds["delta_max"]
preds["value_min"] = preds["value_mean"] - preds["delta_min"]

truths = hh_valid[["time", "value_max", "value_min", "value_mean"]]

print(f"manually calcd: {score_model(preds, truths)}")

# Now let's predict using custom class
custom_ets = ETS(48)
custom_ets.fit(hh_train)
custom_preds = custom_ets.predict(hh_valid[["time", "value_mean"]])
print(f"From class: {score_model(custom_preds, truths)}")

# And then generate August prediction
aug_ets = ETS(48)
aug_ets.fit(hh_data)

aug_hh = load_hh_data("staplegrove").query("type == 'august'")[["time", "value"]]
aug_hh.columns = ["time", "value_mean"]

aug_preds = aug_ets.predict(aug_hh)
create_submissions_file(aug_preds, "august", "ml")
