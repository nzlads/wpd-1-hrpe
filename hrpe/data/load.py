"""
Functions to help load raw data to work with
"""
import os
import glob
import pandas as pd
import datetime


def validate_date(date_text):
    try:
        datetime.datetime.strptime(date_text, '%Y-%m-%d')
        return datetime.datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD")


def check_substation(substation):
    substation = substation.lower()
    valid_stations = ['staplegrove', 'geevor', 'mousehole']
    if (substation not in valid_stations):
        raise Exception(f"substation not in valid list: {valid_stations}")
    return substation


def time_check(time_start, time_end):
    if(time_start is not None):
        time_start = validate_date(time_start)

    if(time_end is not None):
        time_end = validate_date(time_end)

    return time_start, time_end


def filter_data_by_time(data, time_start, time_end):
    if(time_start is not None):
        data = data[data['time'] >= time_start]

    if(time_end is not None):
        data = data[data['time'] < time_end]

    return data


def load_weather(substation, time_start=None, time_end=None):
    pass


def load_hh_data(substation, time_start=None, time_end=None):
    pass


def load_maxmin_data(substation, time_start=None, time_end=None):

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






def load_minute_data(substation, time_start=None, time_end=None):
    """
    Loads the minute data for a given substation
    :param substation: the name of the substation

    load_minute_data('staplegrove', '2020-01-01', '2020-01-02')
    load_minute_data('staplegrove', '2020-01-01')
    load_minute_data('staplegrove')
    load_minute_data('fake_station')
    """

    # whatever
    check_substation(substation)

    # Get times
    time_start, time_end = time_check(time_start, time_end)

    # File path - find minute data
    file_dir = os.path.join("data", "raw", substation)
    if not os.path.exists(file_dir):
        raise Exception("The requested substation does not exist")
    files = glob.glob(rf"{file_dir}/*.csv")

    for fil in files:
        if "minute" in fil:
            filepath = fil

    # Read data
    data = pd.read_csv(filepath, parse_dates=[2, 4, 8])

    # Assert cols
    expected_cols = ['Unnamed: 0', 'attrId', 'maxtime', 'maxvalue', 'mintime', 'minvalue',
                     'quality', 'samplecount', 'time', 'units', 'value']
    col_names_match = data.columns == expected_cols

    if(not all(col_names_match)):
        raise Exception(f"Column names do not match\nGot: {data.columns},\n expected: {expected_cols}")

    # Filter by time
    data = filter_data_by_time(data, time_start, time_end)

    return data
