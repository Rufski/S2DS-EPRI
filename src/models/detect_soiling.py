"""
functions for detecting the cleaning events and obtaining the soiling profile
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from src.features.build_features import find_true_cleaning_events


def detect_cleaning_events_with_rolling_avg(
        power_signal,
        window_size_l=3,
        window_size_r=3,
        window_overlap=1,
        avg_function=np.max,
        cleaning_peak_distance=3,
        cleaning_peak_prominence=0.0125,
        ):

    """
    Detect the cleaning events in a power-signal

    Calculate the cleaning profile by computing the relative difference between
    two rolling window averages with an overlap, cleaning events are identified
    as peaks in this cleaning profile

        Args:
            power_signal (pandas.Series): time-series of the power-signal in
                pandas format
            window_size_l (int, optional): size of the left rolling window,
                defaults to 3 (good value for daily time-series)
            window_size_r (int, optional): size of the right rolling window,
                defaults to 3 (good value for daily time-series)
            window_overlap (int, optional): overlap between the two windows,
                defaults to 1 (good value for daily time-series)
            avg_function (np.function, optional): function for averaging over
                the respective rolling windows, defaults to np.max, can be any
                function that takes a np.array and returns a single float
            cleaning_peak_distance (int, optional): minimal distance between
                two cleaning events in cleaning_profile
            cleaning_peak_prominence (int, optional): minimal prominence of
                the cleaning events in cleaning_profile

        Returns:
            cleaning_profile (np.array): cleaning profile obtained from the
                power signal, i.e., the relative difference between the two
                rolling window averages (which are overlapping by
                window_overlap)
            cleaning_events_index (np.array): index of cleaning events
                identified in the cleaning profile
            cleaning_events_height (np.array): peak height of cleaning events
                identified in the cleaning profile
    """

    # get the left and right rolling window averages
    rolling_avg_l = power_signal.rolling(window_size_l).apply(avg_function)
    rolling_avg_r = power_signal.rolling(window_size_r).apply(avg_function)

    # compute difference btw. left and right rolling window averages
    window_offset = window_size_l - window_overlap
    cleaning_profile = (rolling_avg_r.to_numpy()[window_offset:] -
                        rolling_avg_l.to_numpy()[:-window_offset])

    # normalize w/ respect to the left rolling window
    cleaning_profile /= rolling_avg_l.to_numpy()[window_offset:]

    # find the peaks in the cleaning profile using scipy.signal.find_peaks
    cleaning_events_index = find_peaks(cleaning_profile,
                                       distance=cleaning_peak_distance,
                                       prominence=cleaning_peak_prominence)[0]

    # get peak height of cleaning events
    cleaning_events_height = cleaning_profile[cleaning_events_index]

    # shift by cleaning events by 1 in order to fit to power signal
    cleaning_events_index += 1

    return (cleaning_profile, cleaning_events_index, cleaning_events_height)


def find_soiling_factor_with_rolling_avg(df, cleaning_events_index, cleaning_events_height):
    """
    Detect soiling factor from cleaning events and cleaning gains.

        Args:
            df (pandas.DataFrame): time series of the power signal 
            cleaning_events_index (np.array): index of cleaning events
                identified in the cleaning profile
            cleaning_events_height (np.array): peak height of cleaning events
                identified in the cleaning profile
        Returns:
            pandas.DataFrame: with the index matching the index of df and the
                estimated soiling factor in the column 'soiling_factor'
    """
    slopes = np.array([1.])
    start_of_soiling = df.index[0]
    for i in range(len(cleaning_events_index)):
        end_of_soiling = df.index[cleaning_events_index[i]]
        soiling_duration = (end_of_soiling - start_of_soiling).days
        slopes = np.append(slopes, 1 - np.arange(soiling_duration) * (cleaning_events_height[i]/soiling_duration) )
        slopes[-1] = 1
        start_of_soiling = end_of_soiling
    output_df = pd.DataFrame(index = df.index)
    output_df['soiling_factor'] = np.append(slopes, np.ones( (df.index[-1] - start_of_soiling).days))
    return output_df


def fing_soiling_profile_with_rolling_avg(df):
    """
    Calculate the soiling profile for a power signal.

    The function uses rolling average to detect cleaning events.

        Args:
            df (pandas.DataFrame): time series of the power signal 
        Returns:
            pandas.DataFrame: with the index matching the index of df and the
                estimated soiling factor in the column 'soiling_factor'
    """
    _, cleaning_index, cleaning_heights = detect_cleaning_events_with_rolling_avg(df.Power)
    return find_soiling_factor_with_rolling_avg(df, cleaning_index, cleaning_heights)


def get_accuracy_of_predicted_cleaning_events(
        df,
        prediction=None,
        tolerance_in_days=0,
        amplitude=True):

    """
    Quantify the accuracy of the cleaning event detection

    returns different quantities for the accuracy of the cleaning event
    detection:

        Args:
            df (pandas.DataFrame): dataframe containing the power signal and
                the soiling profile
            prediction (tuple): tuple containing two arrays: indices of the
                predicted cleaning events and their amplitudes, if it is None
                then detect_cleaning_events_with_rolling_avg() is called to
                obtain these two arrays, if amplitude is False then the
                prediction tuple should only contain the indices array
            tolerance_in_days: how close the predicted index has to be to the
                actual cleaning event, defaults to 0
            ampitude(bool): if the cleaning amplitude should also be used in
                the calculation, cleaning amplitude corresponds to a weight of
                the individual cleaning events

        Returns:
            dict_results (dictionary): dictionary containing the results
    """

    if prediction is None:
        profile_predict, index_predict, peakheight_predict = \
            detect_cleaning_events_with_rolling_avg(df['Power'])
    elif prediction is not None and amplitude is True:
        index_predict, peakheight_predict = prediction[0], prediction[1]
    elif prediction is not None and amplitude is False:
        index_predict = prediction[0]

    df_true = find_true_cleaning_events(df)
    index_true = np.where(df_true['cleaning_event'] == 1)[0]

    if amplitude is True:
        peakheight_true = df_true['soiling_loss'].to_numpy()[index_true]

    index_predict_in_true = np.full((index_predict.shape), False, dtype=bool)
    index_true_in_predict = np.full((index_true.shape), False, dtype=bool)

    for index_shift in range(-tolerance_in_days, tolerance_in_days+1):
        index_predict_in_true = np.logical_or(
                np.in1d(index_predict, index_true+index_shift),
                index_predict_in_true)
        index_true_in_predict = np.logical_or(
                np.in1d(index_true+index_shift, index_predict),
                index_true_in_predict)

    n_predict_tot = index_predict.size
    n_true_tot = index_true.size
    n_predict_in_true = np.sum(index_predict_in_true)
    n_true_in_predict = np.sum(index_true_in_predict)

    if amplitude is True:
        explained_amplitude_predict = np.sum(
            peakheight_predict[index_predict_in_true]) / \
            np.sum(peakheight_true)
        explained_amplitude_true = np.sum(
            peakheight_true[index_true_in_predict]) / \
            np.sum(peakheight_true)

    n_true_pos = n_predict_in_true
    n_false_pos = n_predict_tot - n_predict_in_true
    n_false_neg = n_true_tot - n_true_in_predict

    p_cods = n_true_pos / (n_true_pos + n_false_pos)
    r_cods = n_true_pos / (n_true_pos + n_false_neg)
    f1_cods = 2 * r_cods * p_cods / (r_cods+p_cods)

    dict_results = {
        'n_true_pos': n_true_pos,
        'n_false_pos': n_false_pos,
        'n_false_neg': n_false_neg,
        'r_cods': r_cods,
        'p_cods': p_cods,
        'f1_cods': f1_cods,
    }

    if amplitude is True:
        dict_results['explained_amplitude_predict'] = \
            explained_amplitude_predict
        dict_results['explained_amplitude_true'] = \
            explained_amplitude_true

    return dict_results
