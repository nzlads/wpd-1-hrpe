import pandas as pd


class PeriodModel:
    expected_cols = [
        "time",
        "value_max",
        "value_min",
        "value_mean",
        "delta_max",
        "delta_min",
        "period",
    ]

    def check_train_data(self, train: pd.DataFrame):
        """
        Check that the training data fits expectations as required
        """
        missing = [col for col in self.expected_cols if col not in train.columns]
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

    def fit(self, train: pd.DataFrame):
        """
        Fit a model using the training data
        """
        self.check_train_data(train)

        pass

    def predict(self):
        """
        Generate data frame of prediction
        Must return an output that is a data frame with value_max, value_min, value_mean and time columns
        """
        pass
