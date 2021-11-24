import pandas as pd

def calculate_deltas(data):
    """
	input
	data - hh data, use transform.
    
    return hh_data with extra columns:
	delta max - maximum value over the 30 minute perod subtracted by the mean.
	delta min - mean value over the 30 minute perod subtracted by the minimum value.
	range - delta max - delta min
    """
    # Check if time period has been added.
    # If not, run datetime features.

    df = data.copy()

    # Aggregrate to max min and mean
    hh_data["delta_max"] = hh_data["value_max"] - hh_data["value_mean"]
    hh_data["delta_min"] = hh_data["value_mean"] - hh_data["value_min"]
    hh_data['range'] = hh_data["delta_max"] - hh_data["delta_min"]

    return hh_data


 	hh_data["is_weekday"] = hh_data["is_weekday"].astype(int)

def calculate_maxmin(data):
    """
    take  Half hourly data, work out value min and value max
    Add as columns
    return data
    """