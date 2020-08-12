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
    print(clipping_df.shape)
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
    found_non_empyt_window = False

    # To measure at the end how much data is removed
    if verbose:
        initial_size = len(time_series_df)

    for day in range(0, 366):
        clipping_df = time_series_df[
                (day_of_year == day + 1) &
                (power == max_power)
            ]
        if clipping_df.empty:
            windows[0][day] = -1
            windows[1][day] = -1
        else:
            windows[0][day] = clipping_df.minute_of_day.min()
            windows[1][day] = clipping_df.minute_of_day.max()
            found_non_empyt_window = True

    if not found_non_empyt_window:
        # No non empty window was found
        if verbose:
            print("Power never exceeds max value; no data removed.")
        return time_series_df

    # Remove the windows
    time_series_df = time_series_df[
        (minute_of_day < windows[0][day_of_year - 1]) |
        (minute_of_day > windows[1][day_of_year - 1])
        ]

    if verbose:
        print(str(100 * (1 - len(time_series_df)/initial_size)) +
              "% of the data has been removed!")

    return time_series_df
