"""
this module contains functions for normalizing the power signal and obtaining
the performance index
"""

import numpy as np
from pvlib.clearsky import detect_clearsky


def normalize_power_signal(
        data_frame,
        poa_reference=None,
        clipping_threshold=1825.,
        clearsky=True,
        verbose=False):

    """
    normalize the power signal of a time-series

    normalization method is taken from Daniel Fregosi (method he uses for
    rdtools

        Args:
            data_frame (Pandas DataFrame): dataframe of synthetic data
            poa_reference (Pandas Series, optional): clearsky POA signal
                defaults to None (then data_frame.POA = sensor irradiance
                is used)
                defaults to 0, ie, the first timeseries in the dataset
            clipping_threshold (float, optional): threshold for clipping,
                defaults to 1825
            clearsky (bool, optional): if true detect_clearsky() from pvlib is
                used to throw out cloudy datapoints
            verbose (bool, optional): print output if true, defaults to False

        Returns:
            p_norm_daily (Pandas Series): normalized power signal
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
    data_frame_temp = data_frame[clearsky_mask]
    data_frame_temp = data_frame_temp[nighttime_mask]

    # inbuilt number for the clipping threshold
    clipping_mask = data_frame_temp.Power < clipping_threshold

    # throw away cloudy, nighttime, and clipped periods
    p_norm = p_norm[clearsky_mask]
    poa_reference = poa_reference[clearsky_mask]
    p_norm = p_norm[nighttime_mask]
    poa_reference = poa_reference[nighttime_mask]
    p_norm = p_norm[clipping_mask]
    poa_reference = poa_reference[clipping_mask]

    if verbose is True:
        print(f'{p_norm.size/data_size_init:.2f} % of datapoints remaining '
              'after clearsky-detection, nighttime-removal, and clipping-'
              'removal')

    # calculate daily aggregate with POA as weight function
    p_norm_daily = p_norm * poa_reference
    p_norm_daily = p_norm_daily.resample('D').sum()
    p_norm_daily /= poa_reference.resample('D').sum()

    # return normalize power (PI)
    return p_norm_daily

def detect_pi_outliers(
        pi_signal,
        threshold_min=0.70,
        threshold_max=1.00,
        verbose=False):

    """
    detect outliers in pi signal

    default threshold of 0.7 and 1.0 were found empirically

        Args:
            pi_signal (Pandas Series): performance index
            threshold_min (float, optional): lower bound, defaults to 0.7
            threshold_max (float, optional): upper bound, defaults to 1.0
            verbose (bool, optional): print output if true, defaults to False

        Returns:
            mask_outliers (bool array): mask which only leaves the outliers
            n_outliers (int): number of outliers
            ratio_outliers (float): ratio of number of outliers to number
                non-nan datapoints
    """

    signal_size = pi_signal.size
    signal_notna_size = pi_signal.notna().sum()
    n_nans_before = pi_signal.isna().sum()

    if verbose is True:
        print(f'\nPI signal contains {n_nans_before / signal_size * 100.:.2f} '
              '% NaNs\n')

    mask_outliers_above = pi_signal > threshold_max
    mask_outliers_below = pi_signal < threshold_min
    mask_outliers = np.logical_or(mask_outliers_above, mask_outliers_below)

    n_outliers = mask_outliers.sum()
    ratio_outliers = n_outliers / signal_notna_size
    percent_outliers = ratio_outliers * 100.

    if verbose is True:
        print(f'detected {n_outliers:d} outliers (corresponging to '
              '{percent_outliers:.2f} % of non-NaN PI signal)\n')

    return mask_outliers, n_outliers, ratio_outliers
