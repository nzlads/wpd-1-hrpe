import pandas as pd

from darts import TimeSeries
from darts.models import LightGBMModel


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


class DartsLGBMModel(DartsModel):
    """
    Class implementing LGBM model for dmax and dmin
    """

    def _check_lags_dict(self, lags_dict: dict, num_covariates: int = 0):
        assert "lags" in lags_dict.keys(), "Must define lags"

        if num_covariates > 0:
            assert (
                "lags_future_covariates" in lags_dict.keys()
            ), "Must define lags_future_covariates"
            if type(lags_dict["lags_future_covariates"]) == list:
                assert (
                    len(lags_dict["lags_future_covariates"]) == num_covariates
                ), "lags_future_covariates as list must be same length as number of covariates"
                assert all(
                    type(i) == int for i in lags_dict["lags_future_covariates"]
                ), "lags_future_covariates should be list of ints"
            else:
                assert (
                    type(lags_dict["lags_future_covariates"]) == tuple
                ), "If not using list of integers, lags_future_covariates should be a tuple"

    def __init__(
        self,
        dmax_covariates: list = [],
        dmin_covariates: list = [],
        dmax_lags: dict = {"lags": None},
        dmin_lags: dict = {"lags": None},
    ):
        self._check_lags_dict(dmax_lags, num_covariates=len(dmax_covariates))
        self._check_lags_dict(dmin_lags, num_covariates=len(dmin_covariates))

        self.dmax_model = LightGBMModel(**dmax_lags)
        self.dmin_model = LightGBMModel(**dmin_lags)

        self.dmax_uses_lags = any([d is not None for d in dmax_lags.values()])
        self.dmin_uses_lags = any([d is not None for d in dmin_lags.values()])

        self.dmax_covariates = dmax_covariates
        self.dmin_covariates = dmin_covariates

    def fit(self, train: pd.DataFrame):
        self._check_train_data(train)

        # Convert train data to time series for the model fitting
        dmax_ts = self._convert_to_ts(train, "delta_max")
        dmin_ts = self._convert_to_ts(train, "delta_min")

        dmax_cov = self._convert_to_ts(train, self.dmax_covariates)
        dmin_cov = self._convert_to_ts(train, self.dmin_covariates)

        self.dmax_model.fit(dmax_ts, future_covariates=dmax_cov)
        self.dmin_model.fit(dmin_ts, future_covariates=dmin_cov)

    def predict(self, forecast: pd.DataFrame, future_covariates_df: pd.DataFrame):
        dmax_cov = self._convert_to_ts(future_covariates_df, self.dmax_covariates)
        dmin_cov = self._convert_to_ts(future_covariates_df, self.dmin_covariates)

        fc = forecast.copy().set_index("time")
        fc_horizon = len(forecast)

        preds = pd.DataFrame.join(
            self.dmax_model.predict(
                n=fc_horizon, future_covariates=dmax_cov
            ).pd_dataframe(),
            self.dmin_model.predict(
                n=fc_horizon, future_covariates=dmin_cov
            ).pd_dataframe(),
        )
        fc = fc.join(preds)
        fc["value_max"] = fc["value_mean"] + fc["delta_max"]
        fc["value_min"] = fc["value_mean"] - fc["delta_min"]
        return fc.reset_index()[["time", "value_max", "value_min", "value_mean"]]
