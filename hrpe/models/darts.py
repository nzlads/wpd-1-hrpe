import pandas as pd

from darts import TimeSeries


class DartsModel:
    """
    Model class to support use of Darts models to fit and predict dmax and dmin
    """

    def __init__(self, dmax_model, dmin_model):
        self.dmax_model = dmax_model
        self.dmin_model = dmin_model

    def __str__(self):
        return f"Delta Max model: {str(self.dmax_model)}; Delta Min model: {str(self.dmin_model)}"

    def _check_train_data(self, train: pd.DataFrame):
        """
        Check that the training data fits expectations as required
        """
        expected_cols = [
            "time",
            "value_max",
            "value_min",
            "value_mean",
            "delta_max",
            "delta_min",
            "period",
        ]

        missing = [col for col in expected_cols if col not in train.columns]
        assert missing == [], f"The following columns are missing: {missing}"

        expected_dates = pd.date_range(
            start=train["time"].min(), end=train["time"].max(), freq="30T"
        )
        assert len(expected_dates) == len(train), "Some dates are missing"
        assert train[
            "time"
        ].is_monotonic_increasing, "Dates are not strictly increasing"

        period_counts = train["period"].value_counts()
        assert len(period_counts) == 48, "There should be 48 periods"
        assert (
            len(period_counts.unique()) == 1
        ), "The periods in the training data should be the same length"

    def _convert_to_ts(self, df: pd.DataFrame, target: str):
        return TimeSeries.from_dataframe(
            df, time_col="time", value_cols=target, freq="30T"
        )

    def fit(self, train: pd.DataFrame):
        self._check_train_data(train)

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


class BestThetaModel(DartsModel):
    """
    Using Theta, but best value of theta
    """
