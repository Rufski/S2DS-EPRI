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
        nighttime=True,
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

    # calculate clipping mask for throwing away clipped data afterward
    if clipping == 'universal':
        clipping_window_limits = return_universal_clipping_window(data_frame)
    elif clipping == 'flexible':
        clipping_window_limits = return_flexible_clipping_window(data_frame)

    # throw away cloudy periods
    if clearsky is True:
        clearsky_mask = detect_clearsky(data_frame.POA, poa_reference,
                                        data_frame.index, clearsky_window_size)
        data_frame = data_frame[clearsky_mask]
        if verbose is True:
            print('{:.2f} % of data remaining after clearsky '
                  'detection.'.format(data_frame.Power.size / data_size_init))

    # throw away clipping data
    if clipping == 'basic':
        data_frame = data_frame[data_frame.Power < 1827.0]
    elif clipping == 'flexible':
        day_of_year = data_frame.index.dayofyear.to_numpy()
        data_frame = data_frame[
            (data_frame.minute_of_day < clipping_window_limits[0][day_of_year - 1]) |
            (data_frame.minute_of_day > clipping_window_limits[1][day_of_year - 1])
            ]
    elif clipping == 'universal':
        data_frame = data_frame[
            (data_frame.minute_of_day < clipping_window_limits[0]) |
            (data_frame.minute_of_day > clipping_window_limits[1])
            ]
    if (clipping is not None) and (verbose is True):
        print('{:.2f} % of data remaining after clipping '
              'removal.'.format(data_frame.Power.size / data_size_init))

    # throw away nighttime data
    if nighttime is True:
        data_frame = remove_night_time_data(data_frame)
        if verbose is True:
            print('{:.2f} % of data remaining after night-time '
                  'removal.'.format(data_frame.Power.size / data_size_init))

    # calculate expected power and normalize
    if clearsky is True:
        poa_reference = poa_reference[clearsky_mask]
        t_factor = t_factor[clearsky_mask]
    p_expected = poa_reference * p_ref / 1000. * t_factor
    p_norm = data_frame.Power / p_expected

    # calculate daily aggregate with POA as weight function
    p_norm_daily = p_norm * poa_reference
    p_norm_daily = p_norm_daily.resample('D').sum()
    p_norm_daily /= poa_reference.resample('D').sum()

    # remove outliers according to threshold
    p_norm_daily = p_norm_daily[p_norm_daily >= outlier_threshold]

    # return normalize power (PI)
    return p_norm_daily
