"""
this module contains functions for generating the performance index
"""

import numpy as np
import pandas as pd
from pvlib.clearsky import detect_clearsky
from src.data.make_dataset import remove_night_time_data, remove_clipping_with_universal_window, return_universal_clipping_window
from src.features.performance_index import normalize_power_signal
from src.data.import_data import import_df_from_zip_pkl, import_df_info_from_zip
from src.data.make_dataset import downsample_dataframe

data_key = 'soil_weather'
n_samples = 50
path_to_data_raw = '../../data/raw/'
path_to_dataset = '{0:s}/synthetic_{1:s}.zip'.format(path_to_data_raw, data_key)

poa_reference_02 = pd.read_pickle('../../data/raw/clear_sky_CO.pkl', compression='gzip')
t_begin = pd.Timestamp('2015-01-01 00:00:00').tz_localize('Etc/GMT+7')
t_final = pd.Timestamp('2019-12-31 23:59:00').tz_localize('Etc/GMT+7')
datetime_01 = pd.date_range(t_begin, poa_reference_02.index[0], freq='min')[:-1]
datetime_03 = pd.date_range(poa_reference_02.index[-1], t_final, freq='min')[1:]
poa_reference_01 = pd.Series(data=0., index=datetime_01, name='POAcs')
poa_reference_03 = pd.Series(data=0., index=datetime_03, name='POAcs')
poa_reference_co = pd.concat((poa_reference_01, poa_reference_02, poa_reference_03))



for index in range(n_samples):
    df = import_df_from_zip_pkl(path_to_dataset, index)
    
    outlier_threshold = 0.0
    
    if data_key in ['basic', 'soil']:
        clearsky = False
    else:
        clearsky = True
    
    if data_key in ['basic', 'soil']:
        poa_reference = None
    elif data_key in ['soil_weather', 'weather']:
        poa_reference = poa_reference_co
    
    p_norm_daily_clipping_basic = normalize_power_signal(df, poa_reference, clearsky=clearsky, nighttime=True, clipping='basic', verbose=True, outlier_threshold = outlier_threshold)
    p_norm_daily_clipping_flexible = normalize_power_signal(df, poa_reference, clearsky=clearsky, nighttime=True, clipping='flexible', verbose=True, outlier_threshold = outlier_threshold)
    p_norm_daily_clipping_universal = normalize_power_signal(df, poa_reference, clearsky=clearsky, nighttime=True, clipping='universal', verbose=True, outlier_threshold = outlier_threshold)
       
    df_pkl = downsample_dataframe(df)
    df_pkl['PI_clipping_basic'] = p_norm_daily_clipping_basic
    df_pkl['PI_clipping_flexible'] = p_norm_daily_clipping_flexible
    df_pkl['PI_clipping_universal'] = p_norm_daily_clipping_universal
    df_pkl.to_pickle('../../data/raw/new/synthetic_{:s}_pi_daily_{:s}.pkl'.format(data_key, str(index+1).zfill(3)), compression = 'gzip', protocol = 3)

