import pandas as pd


def interpolate_weather(weather_data):
    """
    return 30 min interpolated weather data
    """

    interp_cols = [
        "temperature",
        "solar_irradiance",
        "windspeed_north",
        "windspeed_east",
        "pressure",
        "spec_humidity",
    ]

    # Resample and interpolate 1/2 h data
    weather_data_resampled = list()
    for station in weather_data.station.unique():
        wds = weather_data[(weather_data["station"] == station)]
        wds = wds.set_index(["time"])

        # Need to add a half hour to the last time
        wds = (
            wds.reindex(wds.index.union(wds.index.shift(freq="30min")))
            .resample("30min")
            .mean()
        )

        # interpolate the values
        wds[interp_cols] = wds[interp_cols].interpolate()
        wds["station"] = station

        weather_data_resampled.append(wds)

    weather_data_ridx = pd.concat(weather_data_resampled)

    # Index back to time column
    weather_data_ridx.reset_index(level=0, inplace=True)

    return weather_data_ridx


def add_weather(data, weather):
    """
    return Data with weather features
    return vector of strings of which features were added i.e. ['solar', 'wind', etc]
    """
    pass


def use_all_stations(data: pd.DataFrame):
    """
    Use all the weather values from the stations
    """
    assert "station" in data.columns, "Need the station value"

    station_dfs = []
    for station, df in data.groupby("station"):
        station_df = df.set_index("time").drop(columns="station")
        station_df.columns = [f"{col}_{station}" for col in station_df.columns]
        station_dfs.append(station_df)
    return pd.concat(station_dfs, axis=1).reset_index()
