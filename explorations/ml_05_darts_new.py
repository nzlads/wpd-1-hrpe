import pandas as pd

from hrpe.data.load import load_hh_data, load_maxmin_data, load_weather_data
from hrpe.features.weather import interpolate_weather
from hrpe.features.deltas import calculate_deltas
from hrpe.features.time import make_datetime_features
from hrpe.models.eval import score_model

from hrpe.models.darts import DartsModel
from darts.models import (
    ExponentialSmoothing,
    Theta,
    AutoARIMA,
    RegressionEnsembleModel,
    FFT,
)
from darts.utils.utils import SeasonalityMode

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

# Now compare to some other Darts Models
def eval_darts_models(model_dict, train, test):
    mod = DartsModel(**model_dict)
    print(f"Fitting {mod}")
    mod.fit(train)
    preds = mod.predict(test[["time", "value_mean"]])
    truths = test[["time", "value_max", "value_min", "value_mean"]].reset_index(
        drop=True
    )
    score = score_model(preds, truths)
    print(f"{mod} skill score: {score}")


#%%
model_combos = [
    # ETS
    {
        "dmax_model": ExponentialSmoothing(seasonal_periods=48),
        "dmin_model": ExponentialSmoothing(seasonal_periods=48),
    },
    # Theta
    {
        "dmax_model": Theta(
            seasonality_period=48, season_mode=SeasonalityMode.ADDITIVE
        ),
        "dmin_model": Theta(
            seasonality_period=48, season_mode=SeasonalityMode.ADDITIVE
        ),
    },
    # AutoARIMA - don't do this as it takes ages
    # {
    #     "dmax_model": AutoARIMA(m=48, trace=True),
    #     "dmin_model": AutoARIMA(m=48, trace=True)
    # },
    # FFT
    {"dmax_model": FFT(), "dmin_model": FFT()},
]

#%%
for mod_dict in model_combos:
    eval_darts_models(mod_dict, train, test)

# %%
