import pandas as pd


class MonthSeriesSplit:
    """Generate test and train splits for cross-validation by month.

    Like time-series split, but uses one month of data as the splitting point.
    For each split, the test set is 1 month of data.
    Successive training sets are supersets of those that come before them.


    Parameters
    ----------
    n_splits: int, defualt=5
        Number of splits

    min_train_months: int, default=12
        Minimum number of months in training set
    """

    def __init__(self, n_splits=5, min_train_months=12):
        self.n_splits = n_splits
        self.min_train_months = min_train_months

    def _n_months(self, max_time: pd.Timestamp, min_time: pd.Timestamp):
        return (
            (max_time.year - min_time.year) * 12 + (max_time.month - min_time.month) + 1
        )

    def split(self, data: pd.DataFrame):
        """Generate test and train sets from data

        Parameters
        ----------
        data: pd.DataFrame
            Data to be split

        Yields
        ------
        train: pd.DataFrame
            Training data for given split

        test: pd.DataFrame
            Test set for given split

        """
        assert "time" in data.columns, "data must contain 'time' column."
        n_months = self._n_months(max_time=data.time.max(), min_time=data.time.min())
        months_needed = self.n_splits + self.min_train_months
        assert (
            n_months >= months_needed
        ), f"Data does not have enough months for n_splits and min_train months requested: Has {n_months} but needs {months_needed}"

        test_starts = [
            data.time.max().to_period("M").to_timestamp() - pd.DateOffset(months=k)
            for k in range(self.n_splits)
        ]
        for test_start in sorted(test_starts):
            test_end = test_start + pd.DateOffset(months=1)
            yield (
                data.query("time < @test_start"),
                data.query("time >= @test_start & time < @test_end"),
            )
