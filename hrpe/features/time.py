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
    df["year"] = df["time"].dt.year
    df["is_weekday"] = df["time"].dt.dayofweek < 5
    # Create the period timestamp
    df["period_time"] = data["time"].dt.floor("30T")
    df["day_of_year"] = day_of_year(df)
    df["hh_of_week"] = hh_of_week(df)

    # df["hh_of_week"] = df.apply(lambda x: hh_of_week(x),axis=1)
    #

    return df


def day_of_year(df) -> pd.Series:
    """
    Maps each data row such that 1st Jan = 1, 2nd Jan = 2 ...
    Adjusts yday values for leap years. 29 February is now assigned 59.5 and
    following dates are assigned their original yday minus 1.
    This ensures yday values are consistent with dates across all years.
    """
    REQUIRES = ["time"]

    df["day_of_year"] = df["time"].dt.dayofyear
    # Get leap years vector.
    df["leap_year"] = df["time"].dt.is_leap_year

    # For those days with leap years, have loc statements.
    df.loc[df["leap_year"] & df["day_of_year"] == 60, "day_of_year"] = 59.5

    days_after_leap_year = df.loc[df["leap_year"] & df["day_of_year"] > 60]
    df.loc[days_after_leap_year, "day_of_year"] = (
        df.loc[days_after_leap_year, "day_of_year"] - 1
    )

    return df["day_of_year"]


# def day_of_year(row):
#     """
#     Maps each data row such that 1st Jan = 1, 2nd Jan = 2 ...
#     Adjusts yday values for leap years. 29 February is now assigned 59.5 and
#     following dates are assigned their original yday minus 1.
#     This ensures yday values are consistent with dates across all years.
#     Used as an apply function.
#     """

#     # Edit d

#     # df[dayofyear]
#     # df[year]
#     # df[leapyear]=0/1

#     # Check leap year
#     if row["time"].dt.is_leap_year:
#         # 29th Feb is 59.5
#         if row["time"].dt.dayofyear == 60:
#             return 59.5
#         if row["time"].dt.dayofyear > 60:
#             return row["time"].dt.dayofyear - 1
#         else:
#             return row["time"].dt.dayofyear

#     else:
#         return row["time"].dt.dayofyear


def hh_of_week(df) -> pd.Series:
    """
    Takes the data and adds the period day of week for lag calculation purposes.
    returns a Series.

    period = period of day
    """
    REQUIRES = ["period", "day_of_week"]

    hh_of_week = df["period"] + (df["day_of_week"] - 1) * 48

    return hh_of_week
