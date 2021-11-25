import os


from hrpe.data.load import (
    load_hh_data,
    load_maxmin_data,
    load_minute_data,
    load_weather_data,
)
from hrpe.features.transform import minute_data_to_hh_data
from hrpe.features.weather import interpolate_weather


def main():

    print(os.getcwd())

    # Set Vars
    substation = "staplegrove"

    # Load data using load.py function for staplegrove
    # Load data function

    hh_data = load_hh_data(substation=substation)
    maxmin_data = load_maxmin_data(substation=substation)
    minute_data = load_minute_data(substation=substation)
    weather_data = load_weather_data(substation=substation)

    # Build features / differences
    iweather = interpolate_weather(weather_data)

    min2hh_data = minute_data_to_hh_data(maxmin_data)

    return min2hh_data


# Model
# Naive fit
# naive predict


# Scoring
# RMSE score


# Debugging plots


##
# standards:
# minmax = truth
# hh = halfhour


if __name__ == "__main__":
    # execute only if run as the entry point into the program
    main()
