"""
Functions to create features from datetime stamps
"""

import pandas as pd

def make_datetime_features(data: pd.DataFrame):
    """
    Make the following datetime features:
    - period
    - day_of_week
    - month_of_year
    - is_weekday
    :param data: the dataframe
    Returns the dataframe with date features added
    """
    assert "time" in data.columns, "Assumes that the 'time' column is the column to use"
    # Be safe!
    df = data.copy()

    df["period"] = df["time"].dt.hour * 2 + df["time"].dt.minute // 30 + 1
    df["day_of_week"] = df["time"].dt.dayofweek + 1
    df["month_of_year"] = df["time"].dt.month
    df["is_weekday"] = df["time"].dt.dayofweek < 5
    # Create the period timestamp
    df["period_time"] = data["time"].dt.floor("30T")

    return (df)