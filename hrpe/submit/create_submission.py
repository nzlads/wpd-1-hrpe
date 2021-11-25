from zipfile import ZipFile
from glob import glob
from datetime import datetime
from calendar import monthrange
import os
from shutil import copyfile

import pandas as pd

from hrpe.data.load import filter_data_by_time


def create_submissions_file(
    data: pd.DataFrame, month: str, userinitials: str, year=2021
):
    """[Create a submission file for the HRPE]

    Args:
        data (pd.DataFrame): data must have the columns 'time', 'value_max', 'value_min'
        month ([type]): text desc of a month i.e 'august'
        userinitials ([type]): aa, ml, cl
        year (int, optional): [description]. Defaults to 2021.

    Example:
        from hrpe.data.load import load_maxmin_data
        data = load_maxmin_data('staplegrove')
        create_submissions_file(data, 'july', 'aa')

    Returns:
        [type]: [description]
    """
    userinitials = userinitials.lower()
    assert userinitials in ["aa", "ml", "cl"]

    data_filt = _validate_submissions_data(data, month, year)
    csv_path = _create_submission_csv(data_filt, userinitials)
    zippath = _zip_submissions_file(csv_path)
    return zippath


def _validate_submissions_data(data, month, year):

    assert {"time", "value_max", "value_min"}.issubset(
        data.columns
    ), "The data must have the columns 'time', 'value_max', 'value_min'"

    data = data[["time", "value_max", "value_min"]]

    month_num = datetime.strptime(month, "%B").month
    ndays = monthrange(year, month_num)[1]

    # Generate start and end data of a month
    time_start = datetime(year, month_num, 1)
    time_end = datetime(year, month_num, ndays, 23, 59)

    # Filter by time
    data = filter_data_by_time(data, time_start, time_end)
    assert (
        data.shape[0] == ndays * 48
    ), f"The data must have the same number of rows as the number of days in the mont *48, instead it has {data.shape[0]}"

    return data


def _create_submission_csv(data, userinitials):
    path = "data/submissions"

    # number of files in path directory
    files = glob(rf"{path}/predictions_{userinitials}_*.csv")
    idx = len(files)
    csv_path = f"{path}/predictions_{userinitials}_{idx}.csv"
    data.to_csv(csv_path, index=False)

    return csv_path


def _zip_submissions_file(csv_path):
    # Zip the csv file defined the path and rename it to 'Predictions.zip'
    zip_name = "data/Predictions.zip"
    req_name = "predictions.csv"
    req_path = os.path.join("data/submissions", req_name)

    if os.path.exists(zip_name):
        os.remove(zip_name)

    copyfile(csv_path, req_path)

    with ZipFile(zip_name, "w") as zipf:
        zipf.write(req_path, arcname=req_name)

    print(f"Submissions at {zip_name}")
    os.remove(req_path)

    return zip_name
