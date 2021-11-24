
import seaborn as sns
import matplotlib.pyplot as plt

from hrpe.data.load import load_minute_data
from hrpe.features.time import make_datetime_features


data = load_minute_data("staplegrove")
data = make_datetime_features(data)


# First exploration: how many periods have bad data points?
badpp = data.groupby("period_time").agg(
    {'quality': lambda x: x.isin(["Bad", "Bad Ip"]).sum()})
print(
    f"There are {sum(badpp.quality > 0)} periods with bad data out of {len(badpp)}")
print(f"There are {sum(badpp.quality == 30)} periods that are entirely bad")
# 94 of 30672 periods have some bad data, is 0.3%
# 51 periods have entirely bad data data (all 30 mins are bad)
# 0.3% is not much to drop so just drop it
badpp["is_bad"] = badpp["quality"] > 0
data = data.merge(badpp[["is_bad"]], on="period_time")
data = data[~data["is_bad"]]

# Calculate the half-hourly features data from the minute data
hh_data = data.groupby("period_time").agg({"value": ["max", "min", "mean"]})
hh_data.columns = ['_'.join(col) for col in hh_data.columns.to_flat_index()]
hh_data["time"] = hh_data.index
hh_data = make_datetime_features(hh_data).reset_index(drop=True)

hh_data["delta_max"] = hh_data["value_max"] - hh_data["value_mean"]
hh_data["delta_min"] = hh_data["value_mean"] - hh_data["value_min"]
hh_data["is_weekday"] = hh_data["is_weekday"].astype(int)

# plot delta_max over time
g = sns.FacetGrid(hh_data, col="period", col_wrap=8)
g.map(sns.lineplot, "time", "delta_max")
plt.show()
# Plot shows that the larger deltas tend to occur at 6am to 6pm which makes sense

g = sns.FacetGrid(hh_data, col="period", col_wrap=8)
g.map(sns.lineplot, "time", "delta_min")
plt.show()
# Fairly similar patterns for delta_min

# doens't seem to be much in day of week
g = sns.FacetGrid(hh_data[hh_data["period"] == 24], row="day_of_week")
g.map(sns.lineplot, "time", "delta_max")
plt.show()

# Value mean relationship is kind of weird, I can't quite explain it -
# as value_mean increases (i.e. 30min avg demand increases) the delta_max decreases;
# but is more variable at lower value_mean?
sns.relplot(data=hh_data, x="value_mean", y="delta_max", col="period", col_wrap=8)
sns.relplot(data=hh_data, x="value_mean", y="delta_max")
sns.relplot(data=hh_data, x="value_mean", y="delta_min")
plt.show()

# %%
