"""
Functions to help load raw data to work with
"""

import os
import glob
import pandas as pd

def load_minute_data(substation):
    """
    Loads the minute data for a given substation
    :param substation: the name of the substation
    """
    file_dir = os.path.join("data", "raw", substation)
    if not os.path.exists(file_dir):
        raise Exception("The requested substation does not exist")
    files = glob.glob(rf"{file_dir}/*.csv")

    for fil in files:
        if "minute" in fil:
            filepath = fil

    data = pd.read_csv(filepath, parse_dates=[2, 4, 8])
    return data