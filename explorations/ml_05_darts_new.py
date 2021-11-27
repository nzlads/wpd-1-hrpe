import pandas as pd

from hrpe.data.load import load_hh_data, load_maxmin_data, load_weather_data
from hrpe.features.weather import interpolate_weather
from hrpe.features.deltas import calculate_deltas
from hrpe.features.time import make_datetime_features
from hrpe.models.eval import score_model

from hrpe.models.base import DartsModel
from darts.models import ExponentialSmoothing

SUBSTATION = "staplegrove"

hh_data = load_hh_data(SUBSTATION)
maxmin_data = load_maxmin_data(SUBSTATION)

data = pd.merge(maxmin_data, hh_data, on="time").rename(columns={"value": "value_mean"})
data = calculate_deltas(data)
data = make_datetime_features(data)

train = data.query("type == 'pre_august'")
test = data.query("type == 'august'")

aug_ets = DartsModel(
    ExponentialSmoothing(seasonal_periods=48), ExponentialSmoothing(seasonal_periods=48)
)
aug_ets.fit(train)
aug_preds = aug_ets.predict(test[["time", "value_mean"]])
aug_truths = test[["time", "value_max", "value_min", "value_mean"]].reset_index(
    drop=True
)

score_model(aug_preds, aug_truths)
print(aug_ets)
