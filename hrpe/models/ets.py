import pandas as pd

from darts import TimeSeries
from darts.models import ExponentialSmoothing

from .base import Model


class ETS(Model):
    """
    Using the darts ExponentialSmoothing model, which in turn uses the statsmodels Exponential Smoothing model
    """

    def __init__(self, seasonal_periods):
        self.dmax_model = ExponentialSmoothing(seasonal_periods=seasonal_periods)
        self.dmin_model = ExponentialSmoothing(seasonal_periods=seasonal_periods)

    def _convert_to_ts(self, df: pd.DataFrame, target: str):
        return TimeSeries.from_dataframe(
            df, time_col="time", value_cols=target, freq="30T"
        )

    def fit(self, train: pd.DataFrame):
        self.check_train_data(train)

        # Convert train data to time series for the model fitting
        dmax_ts = self._convert_to_ts(train, "delta_max")
        dmin_ts = self._convert_to_ts(train, "delta_min")

        self.dmax_model.fit(dmax_ts)
        self.dmin_model.fit(dmin_ts)

    def predict(self, forecast: pd.DataFrame):
        """
        Generate predictions for output for dates in forecast
        Expect forecast to have time, value_mean cols
        """
        fc = forecast.copy().set_index("time")
        fc_horizon = len(forecast)
        preds = pd.DataFrame.join(
            self.dmax_model.predict(fc_horizon).pd_dataframe(),
            self.dmin_model.predict(fc_horizon).pd_dataframe(),
        )
        fc = fc.join(preds)
        fc["value_max"] = fc["value_mean"] + fc["delta_max"]
        fc["value_min"] = fc["value_mean"] - fc["delta_min"]
        return fc.reset_index()[["time", "value_max", "value_min", "value_mean"]]
