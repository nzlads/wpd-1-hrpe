import pandas as pd

from hrpe.features.weather import add_weather

data = add_weather(hh_data, weather_data)
xdd(data, 13, 12)


def hdd(data: pd.DataFrame, cutoff: float, temp_col="temperature") -> pd.DataFrame:
    """
    Calculates the HDD (heat degree day) of a dataset.

    :param data: The data to calculate HDD for.
    :param cutoff: The cutoff to use for the HDD calculation.
    :return: The HDD of the data.
    """

    assert {"time", temp_col}.issubset(
        data.columns
    ), f"Make sure you are passing weather data in fool!"

    data["hdd"] = data[temp_col] - cutoff
    data.loc[data["hdd"] < 0, "hdd"] = 0

    return data


def cdd(data: pd.DataFrame, cutoff: float, temp_col="temperature") -> pd.DataFrame:
    """
    Calculates the CDD (cool degree day) of a dataset.

    :param  data The data to calculate CDD for.
    :param cutoff: The cutoff to use for the CDD calculation.
    :return: The CDD of the data.
    """

    assert {"time", temp_col}.issubset(
        data.columns
    ), f"Make sure you are passing weather data in fool!"

    data["cdd"] = cutoff - data[temp_col]
    data.loc[data["cdd"] < 0, "cdd"] = 0

    return data


def xdd(
    data: pd.DataFrame,
    h_cutoff: float = 15,
    c_cutoff: float = 5,
    temp_col="temperature",
) -> pd.DataFrame:
    """
    Calculates the XDDs (X degree day) of a dataset.

    :param data: The data to calculate XDDs for.
    :param h_cutoff: The cutoff to use for the XDDs calculation.
    :param c_cutoff: The cutoff to use for the XDDs calculation.
    :return: The XDDs of the data.
    """

    assert h_cutoff > c_cutoff, f"Cutoffs are wrong way around"

    data = hdd(data, h_cutoff, temp_col)
    data = cdd(data, c_cutoff, temp_col)

    return data
