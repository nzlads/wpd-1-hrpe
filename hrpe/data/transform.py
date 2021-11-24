import pandas as pd
from hrpe.features.time import make_datetime_features
 
def minute_data_to_hh_data(data:pd.DataFrame):
    """
    Transforms the raw minute data into half hour increments.
    """

    df = data.copy()

    if not "period_time" in  df.columns:
        df = make_datetime_features(df)


    hh_data = df.groupby("period_time").agg({"value": ["max", "min", "mean"]})
    hh_data.columns = ['_'.join(col) for col in hh_data.columns.to_flat_index()]
    hh_data["time"] = hh_data.index

    hh_data.reset_index(drop=True)

    return hh_data



