import pandas as pd

def calculate_deltas(data):
    """
	input
	data - hh data, use transform.
    
    return hh_data with extra columns:
	delta max - maximum value over the 30 minute perod subtracted by the mean.
	delta min - mean value over the 30 minute perod subtracted by the minimum value.
	range -  max subtracted by min
    """

    # Check if time period has been added.
    # If not, run datetime features.

    hh_data = data.copy()

    # Calcs delta.
    hh_data["delta_max"] = hh_data["value_max"] - hh_data["value_mean"]
    hh_data["delta_min"] = hh_data["value_mean"] - hh_data["value_min"]
    hh_data['range'] = hh_data["value_max"] - hh_data["value_min"]

    return hh_data


def calculate_maxmin(data):
    """
    take  Half hourly data, Given deltas, work out values.
    Add as columns
    return data
    """

    hh_data = data.copy()

    # Aggregrate to max min and mean
    hh_data["value_max"] = hh_data["delta_max"] + hh_data["value_mean"]
    hh_data["value_min"] = hh_data["value_mean"] - hh_data["delta_min"]

    return hh_data

