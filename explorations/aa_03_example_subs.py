from hrpe.submit.create_submission import create_submission_file
import pandas as pd

# Copy July across
# 1.65
from hrpe.data.load import load_maxmin_data

data = load_maxmin_data("staplegrove")
data2 = data
data2.time = data2.time + pd.DateOffset(months=1)
create_submissions_file(data2, "august", "aa")


# Set baseline
#  1.0
from hrpe.data.load import load_hh_data

data = load_hh_data("staplegrove")
data["value_max"] = data["value"]
data["value_min"] = data["value"]
create_submissions_file(data, "august", "aa")


# Set baseline  with 10% increment
# 1.03! worse?
from hrpe.data.load import load_hh_data

data = load_hh_data("staplegrove")
data["value_max"] = data["value"] * 1.1
data["value_min"] = data["value"] * 0.9
create_submissions_file(data, "august", "aa")


# Work out average percentage diff per period
# 1.0450063459
import numpy as np

hh = load_hh_data("staplegrove")
truth = load_maxmin_data("staplegrove")

hh_tr = hh.query("type == 'pre_august'")
hh_aug = hh.query("type == 'august'")

## Join hh_tr and truth
#
joined = hh_tr.merge(truth, on=["time"], how="left")

max_inc = np.mean(joined.value_max / joined.value)
min_dec = np.mean(joined.value_min / joined.value)
hh_aug.value_max = hh_aug.value * max_inc
hh_aug.value_min = hh_aug.value * min_dec
create_submissions_file(hh_aug, "august", "aa")


# Mac increment dependant on period
# 3.112341241241 cause of negatives
import numpy as np
from hrpe.features.time import make_datetime_features

hh = load_hh_data("staplegrove")
truth = load_maxmin_data("staplegrove")

hh_tr = hh.query("type == 'pre_august'")
hh_aug = hh.query("type == 'august'")


## Join hh_tr and truth
#
joined = hh_tr.merge(truth, on=["time"], how="left")
dtjoin = make_datetime_features(joined)
dthh_aug = make_datetime_features(hh_aug)


dtjoin["max_inc"] = dtjoin["value_max"] / dtjoin["value"]
dtjoin["min_inc"] = dtjoin["value_min"] / dtjoin["value"]

increments = (
    dtjoin.groupby(["period", "day_of_week"])
    .agg({"max_inc": ["mean"], "min_inc": ["mean"]})
    .droplevel(axis=1, level=1)
    .reset_index()
)
subdata = dthh_aug.merge(increments, on=["period", "day_of_week"], how="left")

subdata["value_max"] = subdata["value"] * subdata["max_inc"]
subdata["value_min"] = subdata["value"] * subdata["min_inc"]
create_submissions_file(subdata, "august", "aa")
