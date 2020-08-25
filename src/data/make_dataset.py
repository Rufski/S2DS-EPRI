import pandas as pd
import numpy as np


def remove_night_time_data(df):
    """
    Remove night time entries in df
    
    Input
    -----
        df : pandas dataframe
    
    Output
    ------
        df : pandas dataframe
    """
    night_power = -1
    df          = df[df.Power != night_power]
    return df


def remove_clipping_with_universal_window(
    time_series_df, verbose=False, max_power=None
):
    """Remove clipping from a power signal.

    The function determines a time window within a day when, over the span
    of the time series, each minute goes through at least one day that hits
    the maximum power beyond which power is maxed out.

    Args:
        time_series_df (Pandas DataFrame): with a time stamp (datetime) as an
            index, a "Power" column (float), and a "minute_of_day" (int) column
        verbose (bool): Optional; print additional info if true
        max_power (float): Optional; specify maximal power

    Returns:
        Pandas DataFrame: a copy of the input DataFrame with the time window
            removed.
    """
    if not max_power:
        max_power = time_series_df.Power.max()
        if verbose:
            print("Max power set as "+str(max_power))

    # To measure at the end how much data is removed
    if verbose:
        initial_size = len(time_series_df)

    clipping_df = time_series_df[time_series_df.Power >= max_power]
    if clipping_df.empty:
        if verbose:
            print("Power never exceeds max value; no data removed.")
        return time_series_df
    else:
        window_first_minute = clipping_df.minute_of_day.min()
        window_last_minute = clipping_df.minute_of_day.max()

    # Removing the fixed window
    time_series_df = time_series_df[
        (time_series_df.minute_of_day < window_first_minute) |
        (time_series_df.minute_of_day > window_last_minute)
        ]

    if verbose:
        print(
            str(100 * (1 - len(time_series_df)/initial_size)) +
            "% of the data has been removed!"
        )

    return time_series_df


def return_universal_clipping_window(
    time_series_df, verbose=False, max_power=None
):
    """return universal clipping window of power signal.

    The function determines a time window within a day when, over the span
    of the time series, each minute goes through at least one day that hits
    the maximum power beyond which power is maxed out.

    Args:
        time_series_df (Pandas DataFrame): with a time stamp (datetime) as an
            index, a "Power" column (float), and a "minute_of_day" (int) column
        verbose (bool): Optional; print additional info if true
        max_power (float): Optional; specify maximal power

    Returns:
        Tupel: limits of the universal window.
    """
    if not max_power:
        max_power = time_series_df.Power.max()
        if verbose:
            print("Max power set as "+str(max_power))

    # To measure at the end how much data is removed
    initial_size = len(time_series_df)

    clipping_df = time_series_df[time_series_df.Power >= max_power]
    if clipping_df.empty:
        if verbose:
            print("Power never exceeds max value; no data removed.")
        return time_series_df
    else:
        window_first_minute = clipping_df.minute_of_day.min()
        window_last_minute = clipping_df.minute_of_day.max()

    # Removing the fixed window
    time_series_df = time_series_df[
        (time_series_df.minute_of_day < window_first_minute) |
        (time_series_df.minute_of_day > window_last_minute)
        ]

    if verbose:
        print(
            str(100 * (1 - len(time_series_df)/initial_size)) +
            "% of the data has been removed!"
        )

    return (window_first_minute, window_last_minute)


def remove_clipping_with_flexible_window(
    time_series_df, verbose=False, max_power=None
):
    """Remove clipping from a power signal.

    The function determines a time window for each day <day> of the year, so
    that for every minute <min> in the time window there is year in the time
    series such that <min>-th minute of the <day>-th day of that year hits the
    maximum power.

    Args:
        time_series_df (Pandas DataFrame): with a time stamp (datetime) as an
            index, a "Power" column (float), and a "minute_of_day" (int) column
        verbose (bool): Optional; print additional info if true
        max_power (float): Optional; specify maximal power

    Returns:
        Pandas DataFrame: a copy of the input DataFrame with the time window
            removed.
    """
    if not max_power:
        max_power = time_series_df.Power.max()
        if verbose:
            print("Max power set as "+str(max_power))

    day_of_year = np.array(time_series_df.index.dayofyear)
    minute_of_day = np.array(time_series_df.minute_of_day)
    power = np.array(time_series_df.Power)
    windows = np.zeros((2, 366))
    found_non_empty_window = False

    # To measure at the end how much data is removed
    if verbose:
        initial_size = len(time_series_df)

    for day in range(0, 366):
        clipping_df = time_series_df[
                (day_of_year == day + 1) &
                (power == max_power)
            ]
        if clipping_df.empty:
            windows[0][day] = -2
            windows[1][day] = -1
        else:
            windows[0][day] = clipping_df.minute_of_day.min()
            windows[1][day] = clipping_df.minute_of_day.max()
            found_non_empty_window = True

    if not found_non_empty_window:
        # no non empty window was found
        if verbose:
            print("power never exceeds max value; no data removed.")
        return time_series_df

    # remove the windows
    time_series_df = time_series_df[
        (minute_of_day < windows[0][day_of_year - 1]) |
        (minute_of_day > windows[1][day_of_year - 1])
        ]

    if verbose:
        print(str(100 * (1 - len(time_series_df)/initial_size)) +
              "% of the data has been removed!")

    return time_series_df


def return_flexible_clipping_window(
    time_series_df, verbose=False, max_power=None
):
    """return universal clipping window of power signal.

    The function determines a time window for each day <day> of the year, so
    that for every minute <min> in the time window there is year in the time
    series such that <min>-th minute of the <day>-th day of that year hits the
    maximum power.

    Args:
        time_series_df (Pandas DataFrame): with a time stamp (datetime) as an
            index, a "Power" column (float), and a "minute_of_day" (int) column
        verbose (bool): Optional; print additional info if true
        max_power (float): Optional; specify maximal power

    Returns:
        windows (list): a list containing two lists: one list with the lower
            bounds and one list with the upper bounds of the windows of each
            day in a year
    """
    if not max_power:
        max_power = time_series_df.Power.max()
        if verbose:
            print("Max power set as "+str(max_power))

    day_of_year = np.array(time_series_df.index.dayofyear)
    minute_of_day = np.array(time_series_df.minute_of_day)
    power = np.array(time_series_df.Power)
    windows = np.zeros((2, 366))
    found_non_empty_window = False

    # To measure at the end how much data is removed
    if verbose:
        initial_size = len(time_series_df)

    for day in range(0, 366):
        clipping_df = time_series_df[
                (day_of_year == day + 1) &
                (power == max_power)
            ]
        if clipping_df.empty:
            windows[0][day] = -2
            windows[1][day] = -1
        else:
            windows[0][day] = clipping_df.minute_of_day.min()
            windows[1][day] = clipping_df.minute_of_day.max()
            found_non_empty_window = True

    if not found_non_empty_window:
        # no non empty window was found
        if verbose:
            print("power never exceeds max value; no data removed.")
        return time_series_df

    return windows

def downsample_dataframe(df,
                         offset='D',
                         clip_method=None,
                         night_method=None,
                         power_sampling_function=np.mean,
                         loss_sampling_function='last'):
    """
    Downsamples the dataframe.

    Downsamples the DataFrame using inbuild resample function of pandas. The
    function has options for removing clipping and night time data. The
    argument <night_methods> can be set to None or 'basic' to either not remove
    any nighttime data or remove all of it, respectively.  The argument
    <clip_method> can be set to None, 'universal', or 'flexible'. The methods
    remove no clipping data, remove clipping data with unversal window, or
    remove clipping data with flexible window respectively. If the optional
    arguments are not speficied, the function downsample to days with no data
    removal.

    A further optional parameters allows two functions to be specified --
    <power_sampling_function> is a function used to aggregate Power column of
    the DataFrame, and <loss_sampling_function> is function used to aggregate
    Degradation, Soiling, and Faults. By default numpy mean is used for the
    <power_sampling_function> and the last value is used for the
    <loss_sampling_function>.

        Args:
            df (pandas.DataFrame): a DataFrame with timedate as index
            offset (str): Optional; offset string representing target downsampling
            clip_method (str): Optional; string representing clipping removal
                method
            night_method (str): Optional; string representing night removal method
            power_sampling_function (function): Optional; function to aggregate the
                Power column
            loss_sampling_function (function): Optional; function to aggregate the
                Degradation, Soiling, Faults
        Returns:
            pandas.DataFrame: downsampled dataframe
    """
    if night_method == 'basic':
        df = remove_night_time_data(df)
    elif night_method is not None:
        print("Error: night_method needs to be None, or 'basic'")

    if clip_method == 'universal':
        df = remove_clipping_with_universal_window(df)
    elif clip_method == 'flexible':
        df = remove_clipping_with_flexible_window(df)
    elif clip_method is not None:
        print(("Error: clip_method needs to be None, 'universal', or \
                'flexible'"))

    return df.resample(offset).agg({'Power': power_sampling_function,
                                    'Degradation': loss_sampling_function,
                                    'Soiling': loss_sampling_function,
                                    'Faults': loss_sampling_function})


def subsamples_generator(dataset, sub_number=100, sub_size_fraction=0.6):

    """
    Generates multiple samples of smaller size from one time series sample.

    Takes in a time series of type Pandas DataFrame, NumPy array or list.
    Selects a specified number of continuous subsets of this time series,
    the subsets having a length that is a specified fraction of the input
    time series. Each subset is chosen to start at some random point of the
    input time series. Outputs a list of DataFrames or lists (depending on
    the input) of dimensions (number of subsamples, length of one subsample)
    indexing the nth subsample on row n.

    Args:
        dataset (Pandas DataFrame, NumPy array or list): the time series from
        which the subsamples will be selected. If a Pandas DataFrame, the
        selection will be done along the index. If a list or NumPy array, the
        object must have dimensions (1, length of time series).
        sub_number (int): number of subsamples to output. Has to be greater
        than 0. Default value: 100.
        sub_size_fraction (float): fraction of the initial time series that
        will correspond to the length of each subsample. Has to be strictly
        between 0.0 and 1.0. Default value: 0.6


    Returns:
        Returns a list containing all the subsamples in NumPy array (when
        input is NumPy array or list) or pandas DataFrame if such was the
        input format.
    """

    np.random.seed(42)

    if sub_number < 1:
        print("Error: requested number of subsamples is not strictly " +
              "positive.")
        return
    if sub_size_fraction >= 1.0 or sub_size_fraction <= 0.0:
        print("Error: requested ratio of subsample size to original dataset " +
              "size is not between 0.0 and 1.0.")
        return
    sub_len = int(len(dataset) * sub_size_fraction)
    new_list = []
    for i in range(0, sub_number):
        first_index = int(np.random.uniform(0, len(dataset)-sub_len))
        new_list.append(dataset[first_index:first_index+sub_len])
    return new_list
