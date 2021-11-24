import pandas as pd

from .base import PeriodModel

class SnaivePeriodModel(PeriodModel):
    """
    A basic naive model that fits model per period
    """
    def fit(self, train: pd.DataFrame, seasonalities: dict):
        """
        Define some expectations for what to expect in the data frames
        """
        self.check_train_data(train)
        self.delta_data = train[["time", "delta_max", "delta_min"]]
        self.seasonalities = seasonalities
    
    def predict(self, forecast: pd.DataFrame):
        """
        Generate predictions for output for dates in forecast
        Expect forecast to have time, value_mean cols
        """
        forecast["source_time"] = forecast["time"] - pd.DateOffset(**self.seasonalities)
        forecast = forecast.merge(self.delta_data, left_on="source_time", right_on="time", suffixes=[None, "_hist"])
        forecast["value_max"] = forecast["value_mean"] + forecast["delta_max"]
        forecast["value_min"] = forecast["value_mean"] - forecast["delta_min"]
        return forecast[["time", "value_max", "value_min", "value_mean"]]