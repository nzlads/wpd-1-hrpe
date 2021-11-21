# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import arrow
import datetime


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
hh_p_aug = pd.read_csv(fs_hh_p_aug)
hh_p_sep = pd.read_csv(fs_hh_p_sep)

# %%
mm_p_preaug = pd.read_csv(fs_mm_p_preaug)
minute_p_preaug = pd.read_csv(fs_minute_p_preaug)


# %%

hh_p_preaug["type"] = 'training'
hh_p_aug["type"] = 'validation'
hh_p_sep["type"] = 'test'


# %%
hh_p = pd.concat([hh_p_preaug, hh_p_aug, hh_p_sep]).reset_index()
hh_p['time'] = pd.to_datetime(hh_p['time'], format='%Y-%m-%d %H:%M:%S')


# %%

# Plot the responses for different events and regions
ax = sns.lineplot(x="time", y="value",
                  hue="type",
                  data=hh_p)
ax.set_xlim(hh_p['time'].min(), hh_p['time'].max())
plt.xticks(rotation=15)
plt.title('Raw Data')
plt.show()
# %%
hh_p_preaug.time = pd.to_datetime(hh_p_preaug.time)
mm_p_preaug.time = pd.to_datetime(mm_p_preaug.time)

# join data from the hh_p and the mm_p_preaug data frames by the time column
plot_data = pd.merge(
    hh_p_preaug,
    mm_p_preaug,
    how='outer',
    on='time'
).melt(
    id_vars=['time', 'type'],
    var_name='name',
    value_name='values'
)
# %%


# plot_data.plot(x='time', y=['value', 'max', 'min'])


sns.relplot(
    data=plot_data.query('time >= "2021-06-01"'),
    x="time", y="values",
    hue="name",
    kind="line"
    # size="choice", col="align", size_order=["T1", "T2"], palette=palette,
    # height=5, aspect=.75, facet_kws=dict(sharex=False),
)


# %%
