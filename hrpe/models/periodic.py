import pandas as pd

from .base import PeriodModel


class SnaivePeriodModel(PeriodModel):
    """
    A basic naive model that fits model per period
    """

    def __init__(self, seasonalities: dict):
        allowed_keys = ["years"]
        assert all(
            [k in allowed_keys for k in seasonalities.keys()]
        ), f"Allowed keys are {allowed_keys}"
        self.seasonalities = seasonalities

    def fit(self, train: pd.DataFrame):
        """
        Define some expectations for what to expect in the data frames
        """
        self.check_train_data(train)
        self.delta_data = train[["time", "delta_max", "delta_min"]]

    def predict(self, forecast: pd.DataFrame):
        """
        Generate predictions for output for dates in forecast
        Expect forecast to have time, value_mean cols
        """
        fc = forecast.copy()
        fc["source_time"] = fc["time"] - pd.DateOffset(**self.seasonalities)
        fc = fc.merge(
            self.delta_data,
            left_on="source_time",
            right_on="time",
            suffixes=[None, "_hist"],
        )
        fc["value_max"] = fc["value_mean"] + fc["delta_max"]
        fc["value_min"] = fc["value_mean"] - fc["delta_min"]
        return fc[["time", "value_max", "value_min", "value_mean"]]
