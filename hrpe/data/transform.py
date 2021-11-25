import pandas as pd
from hrpe.features.time import make_datetime_features


def minute_data_to_hh_data(data: pd.DataFrame):
    """
    Transforms the raw minute data into half hour increments.
    """

    df = data.copy()

    if not "period_time" in df.columns:
        df = make_datetime_features(df)

    # hh_data = df.groupby("period_time").agg({"value": ["max", "min", "mean"],"period": ['first'],"day_of_week":['first'],"month_of_year":['first'],"year":['first']})
    hh_data = df.groupby("period_time").agg({"value": ["max", "min", "mean"]})
    hh_data.columns = ["_".join(col) for col in hh_data.columns.to_flat_index()]
    # Dealing with non-value columns.
    hh_period_data = df.groupby("period_time").first()[
        ["period", "day_of_week", "month_of_year", "year"]
    ]

    hh_data = hh_data.join(hh_period_data)

    hh_data["time"] = hh_data.index

    hh_data.reset_index(drop=True, inplace=True)

    return hh_data
