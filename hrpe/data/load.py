"""
Functions to help load raw data to work with
"""
import os
import glob
import pandas as pd
import datetime
import re


def validate_date(date_text: str):
    """
    Check a date, convert to a datetime object

    :param date_text: A string of the form YYYY-MM-DD
    :returns: A python datetime object 
    :raises keyError: identifies if it doesn't match the format - raises error
    """

    try:
        datetime.datetime.strptime(date_text, '%Y-%m-%d')
        return datetime.datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD")


def check_substation(substation: str):
    """
    Check a substation belongs to valid substations

    :param substation: A string in the following list ['staplegrove', 'geevor', 'mousehole']
    :returns: substation 
    :raises keyError: identifies if it doesn't match the list
    """
    substation = substation.lower()
    valid_stations = ['staplegrove', 'geevor', 'mousehole']
    
    assert substation in valid_stations, f"substation not in valid list: {valid_stations}"

    return substation


def time_check(time_start, time_end):
    """Helper function over the validation of time_start and time_end

    Args:
        time_start ([str]): [start time to filter by]
        time_end ([str]): [end time to filter by]

    Returns:
        [list(dt, dt)]: [packed list of start and end times]
    """

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
    """
    Loads the half hourly data for a given substation
    :param substation: the name of the substation
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

    filepath = list()
    for fil in files:
        if "half_hourly_real_power" in fil:
            filepath.append(fil)

    data_list = list()
    for fl in filepath:
        # Get date in path
        file_type = re.search('power_MW_(.+?).csv', fl)
        if file_type is None:
            file_type = 'all'
        else:
            file_type = file_type.group(1)

        # Read data, add type column
        pddata = pd.read_csv(fl, parse_dates=[0])
        pddata['type'] = file_type
        data_list.append(pddata)

    data = pd.concat(data_list)

    # Assert cols
    expected_cols = ['time', 'value', 'type']
    col_names_match = data.columns == expected_cols

    if(not all(col_names_match)):
        raise Exception(f"Column names do not match\nGot: {data.columns},\n expected: {expected_cols}")

    # Filter by time
    data = filter_data_by_time(data, time_start, time_end)

    return data


def load_maxmin_data(substation, time_start=None, time_end=None):

    """
    Loads the maxmin data for a given substation
    :param substation: the name of the substation
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
        if "max_min" in fil:
            filepath = fil

    # Read data
    data = pd.read_csv(filepath, parse_dates=[0])

    # Assert cols
    expected_cols = ['time', 'value_max', 'value_min']
    col_names_match = data.columns == expected_cols

    if(not all(col_names_match)):
        raise Exception(f"Column names do not match\nGot: {data.columns},\n expected: {expected_cols}")

    # Filter by time
    data = filter_data_by_time(data, time_start, time_end)

    return data

    for fil in files:
        if "minute" in fil:
            filepath = fil

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
