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
    """
    This function determines a time window within a day when, over the span
    of the time series, each minute goes through at least one day that hits
    the maximum power beyond which power is maxed out.

    Input: a time series with a time stamp in datetime format as an index,
    a "Power" column where the power measured is stored as a float and a
    "minute_of_day" column mentioning the nth minute of day the power is
    measured at.

    Output: same time series where the aforementioned time window has been
    removed.
    """
    if not max_power:
        max_power = time_series_df.Power.max()
        if verbose:
            print("Max power set as "+str(max_power))

    window_first_minute = -1
    window_last_minute = -1

    # To measure at the end how much data is removed
    if verbose:
        initial_size = len(time_series_df)

    for minute_of_day in range(0, 1440):

        # If no window first minute found in morning and no window last minute
        # found in afternoon, that means the power never exceeds max_power,
        # so no clipping
        if (minute_of_day == 721 and window_first_minute < 0 and
                window_last_minute < 0):
            if verbose:
                print("Power never exceeds max value; no data removed.")
            return time_series_df

        if verbose and minute_of_day % 100 == 0:
            print("Looking at the "+str(minute_of_day)+"th minute of the day")

        if (window_first_minute < 0 and max_power in list(
                time_series_df[
                    time_series_df.minute_of_day == minute_of_day
                    ].Power
                )):
            window_first_minute = minute_of_day

        if (window_last_minute < 0 and max_power in list(
                time_series_df[
                    time_series_df.minute_of_day == 1439-minute_of_day
                    ].Power
                )):
            window_last_minute = 1439-minute_of_day

        if window_last_minute >= 0 and window_first_minute >= 0:
            break

    # Removing the fixed window
    time_series_df = time_series_df[
        time_series_df.minute_of_day.apply(
            lambda x: x < window_first_minute or x > window_last_minute
        )
    ]

    if verbose:
        print(
            str(100 * (1 - len(time_series_df)/initial_size)) +
            "% of the data has been removed!"
        )

    return time_series_df


def remove_clipping_with_universal_window_faster(
    time_series_df, verbose=False, max_power=None
):
    """
    This function determines a time window within a day when, over the span
    of the time series, each minute goes through at least one day that hits
    the maximum power beyond which power is maxed out.

    This version of the function is expected to be faster and the original.

    Input: a time series with a time stamp in datetime format as an index,
    a "Power" column where the power measured is stored as a float and a
    "minute_of_day" column mentioning the nth minute of day the power is
    measured at.

    Output: same time series where the aforementioned time window has been
    removed.
    """
    if not max_power:
        max_power = time_series_df.Power.max()
        if verbose:
            print("Max power set as "+str(max_power))

    window_first_minute = -1
    window_last_minute = -1

    # To measure at the end how much data is removed
    if verbose:
        initial_size = len(time_series_df)

    for minute_of_day in range(0, 1440):

        # If no window first minute found in morning and no window last minute
        # found in afternoon, that means the power never exceeds max_power,
        # so no clipping
        if (minute_of_day == 721 and window_first_minute < 0 and
                window_last_minute < 0):
            if verbose:
                print("Power never exceeds max value; no data removed.")
            return time_series_df

        if verbose and minute_of_day % 100 == 0:
            print("Looking at the "+str(minute_of_day)+"th minute of the day")

        if (window_first_minute < 0 and max_power in list(
                time_series_df[
                    time_series_df.minute_of_day == minute_of_day
                    ].Power
                )):
            window_first_minute = minute_of_day

        if (window_last_minute < 0 and max_power in list(
                time_series_df[
                    time_series_df.minute_of_day == 1439-minute_of_day
                    ].Power
                )):
            window_last_minute = 1439-minute_of_day

        if window_last_minute >= 0 and window_first_minute >= 0:
            break

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


def remove_clipping_with_universal_window_fastest(
    time_series_df, verbose=False, max_power=None
):
    """
    This function determines a time window within a day when, over the span
    of the time series, each minute goes through at least one day that hits
    the maximum power beyond which power is maxed out.

    This version of the function is expected to be even faster than the
    original.

    Input: a time series with a time stamp in datetime format as an index,
    a "Power" column where the power measured is stored as a float and a
    "minute_of_day" column mentioning the nth minute of day the power is
    measured at.

    Output: same time series where the aforementioned time window has been
    removed.
    """
    if not max_power:
        max_power = time_series_df.Power.max()
        if verbose:
            print("Max power set as "+str(max_power))

    # To measure at the end how much data is removed
    if verbose:
        initial_size = len(time_series_df)

    clipping_df = time_series_df[time_series_df.Power == max_power]
    if clipping_df.empty:
        if verbose:
            print("Power never exceeds max value; no data removed.")
        return time_series_df
    else:
        window_first_minute = clipping_df.Power.min()
        window_last_minute = clipping_df.Power.max()

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
