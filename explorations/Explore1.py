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
minute_p_preaug = os.path.join(dir_data_stap, "MW_Staplegrove_CB905_MW_minute_real_power_MW_pre_august.csv")
mm_p_preaug = os.path.join(dir_data_stap, "MW_Staplegrove_CB905_MW_target_variable_half_hourly_max_min_real_power_MW_pre_august.csv")
fs_hh_p_preaug = os.path.join(dir_data_stap, "MW_Staplegrove_CB905_MW_observation_variable_half_hourly_real_power_MW_pre_august.csv")
fs_hh_p_aug = os.path.join(dir_data_stap, "MW_Staplegrove_CB905_MW_observation_variable_half_hourly_real_power_MW_august.csv")
fs_hh_p_sep = os.path.join(dir_data_stap, "MW_Staplegrove_CB905_MW_observation_variable_half_hourly_real_power_MW_september.csv")

# %%
hh_p_preaug = pd.read_csv(fs_hh_p_preaug)
hh_p_aug = pd.read_csv(fs_hh_p_aug)
hh_p_sep = pd.read_csv(fs_hh_p_sep)
# %%

type(hh_p_preaug.iat[0, 0])
hh_p_preaug['time'] = pd.to_datetime(hh_p_preaug['time'], format='%Y-%m-%d %H:%M:%S')

# %%

sns.lineplot(x="time", y="value", data=hh_p_preaug)
plt.xticks(rotation=15)
plt.title('seaborn-matplotlib example')
plt.show()


# %%
