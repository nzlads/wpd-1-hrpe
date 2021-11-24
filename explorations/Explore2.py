# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import arrow
from datetime import datetime, timedelta, fromisoformat


# %%

# %%

dir_data_raw = '..\\data\\raw'
dir_data_geev = os.path.join(dir_data_raw, 'geevor')
dir_data_mous = os.path.join(dir_data_raw, 'mousehole')
dir_data_stap = os.path.join(dir_data_raw, 'staplegrove')

# %%
fs_geev = os.listdir(dir_data_geev)
fs_mous = os.listdir(dir_data_mous)
fs_stap = os.listdir(dir_data_stap)

# %%
fs_minute_p_preaug = os.path.join(dir_data_stap, "MW_Staplegrove_CB905_MW_minute_real_power_MW_pre_august.csv")
fs_mm_p_preaug = os.path.join(dir_data_stap, "MW_Staplegrove_CB905_MW_target_variable_half_hourly_max_min_real_power_MW_pre_august.csv")
fs_hh_p_preaug = os.path.join(dir_data_stap, "MW_Staplegrove_CB905_MW_observation_variable_half_hourly_real_power_MW_pre_august.csv")
fs_hh_p_aug = os.path.join(dir_data_stap, "MW_Staplegrove_CB905_MW_observation_variable_half_hourly_real_power_MW_august.csv")
fs_hh_p_sep = os.path.join(dir_data_stap, "MW_Staplegrove_CB905_MW_observation_variable_half_hourly_real_power_MW_september.csv")

# %%
hh_p_preaug = pd.read_csv(fs_hh_p_preaug)
mm_p_preaug = pd.read_csv(fs_mm_p_preaug)
minute_p_preaug = pd.read_csv(fs_minute_p_preaug)

# %%
hh = hh_p_preaug
mm = mm_p_preaug
sx = minute_p_preaug

# %%
hh.time = pd.to_datetime(hh.time)
mm.time = pd.to_datetime(mm.time)
sx.time = pd.to_datetime(sx.time)
sx.maxtime = pd.to_datetime(sx.maxtime)
sx.mintime = pd.to_datetime(sx.mintime)


# %%
t1 = '2019-11-01 00:00:00'
t2 = '2019-11-01 01:00:00'
hh = hh_p_preaug.query("time <= @t2")
mm = mm_p_preaug.query("time <= @t2")
sx = minute_p_preaug.query("time <= @t2")


# %%
# Function to take a time and convert it to a half hour between 1 and 48 index

def time_to_half_hour_index(time):
    # Convert string to datetime object
    time = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    # Convert datetime object to a half hour index
    half_hour_index = (time.hour) * 2 + (time.minute >= 30) + 1
    return half_hour_index


time_to_half_hour_index('2019-11-02 00:00:00')

# %%


ax = plt.step(sx.time, sx.value, where='post')
plt.plot(sx.maxtime, sx.maxvalue, 'x', color='black')
plt.plot(sx.mintime, sx.minvalue, 'o', color='black')
plt.xticks(rotation=45)

# %%
# Generate every minute between two times
# from datetime import datetime, timedelta

# def datetime_range(start, end, delta):
#     current = start
#     while current < end:
#         yield current
#         current += delta

# dts = [dt.strftime('%Y-%m-%d T%H:%M Z') for dt in
#        datetime_range(datetime.fromisoformat(t1), datetime.fromisoformat(t2),
#        timedelta(minutes=1))]

# print(dts)
# # xposition = [pd.to_datetime('2019-11-01 00:00:00'), pd.to_datetime('2019-11-01 00:06:00')]


# %%

sns.relplot(
    data=plot_data.query('time >= "2021-07-28"'),
    x="time", y="values",
    hue="name",
    kind="line"
    # size="choice", col="align", size_order=["T1", "T2"], palette=palette,
    # height=5, aspect=.75, facet_kws=dict(sharex=False),
)
