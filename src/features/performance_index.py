"""
this module contains functions for normalizing the power signal and obtaining
the performance index
"""

import numpy as np
from pvlib.clearsky import detect_clearsky
from src.data.make_dataset import remove_night_time_data
from src.data.make_dataset import remove_clipping_with_universal_window
from src.data.make_dataset import return_universal_clipping_window
from src.data.make_dataset import return_flexible_clipping_window


def normalize_power_signal(
        data_frame,
        poa_reference=None,
        clearsky=True,
        clipping='basic',
        outlier_threshold=0.0,
        verbose=False):

    """
    test
    """

    # initialize constant parameters for the calculation
    p_ref = 2772.
    gamma = -0.0045
    t_cell_a = -3.56
    t_cell_b = -0.075
    t_cell_c = 3.
    t_cell_shift = 25.
    clearsky_window_size = 15
    data_size_init = data_frame.Power.size
    if poa_reference is None:
        poa_reference = data_frame.POA

    # calculate temperature factor
    t_cell = np.exp(t_cell_a + t_cell_b * data_frame.Wind.to_numpy()) *\
        data_frame.POA.to_numpy() + data_frame.Tamb.to_numpy() + t_cell_c *\
        data_frame.POA.to_numpy() / 1000.
    t_factor = 1 + gamma * (t_cell - t_cell_shift)

    # calculate expected power and normalize
    p_expected = poa_reference * p_ref / 1000. * t_factor
    p_norm = data_frame.Power / p_expected

    # throw away cloudy periods
    if clearsky is True:
        clearsky_mask = detect_clearsky(data_frame.POA, poa_reference,
                                        data_frame.index, clearsky_window_size)
    else:
        clearsky_mask = np.full(data_frame.POA.size, True)

    # throw away nighttime data
    nighttime_mask = data_frame[clearsky_mask].Power != -1.

    # calculate clipping mask for throwing away clipped data afterward
    if clipping == 'basic':
        data_frame_temp = data_frame[clearsky_mask]
        data_frame_temp = data_frame_temp[nighttime_mask]
        clipping_mask = data_frame_temp.Power < 1800.0

    # throw away cloudy, nighttime, and clipped periods
    p_norm = p_norm[clearsky_mask]
    poa_reference = poa_reference[clearsky_mask]
    p_norm = p_norm[nighttime_mask]
    poa_reference = poa_reference[nighttime_mask]
    p_norm = p_norm[clipping_mask]
    poa_reference = poa_reference[clipping_mask]

    if verbose is True:
        print(f'{p_norm.size/data_size_init:.2f} % of data remaining after '
              'clearsky-detection, nighttime-removal, and clipping-removal')

    # calculate daily aggregate with POA as weight function
    p_norm_daily = p_norm * poa_reference
    p_norm_daily = p_norm_daily.resample('D').sum()
    p_norm_daily /= poa_reference.resample('D').sum()

    # remove outliers according to threshold
    p_norm_daily = p_norm_daily[p_norm_daily >= outlier_threshold]

    # return normalize power (PI)
    return p_norm_daily
