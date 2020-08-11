"""
module for data extraction functions
"""

import os
import zipfile as zf
import time
import pandas as pd


def extract_df_from_zip(path_to_zip, csv_index):

    """
    Takes a dataset in zip-format and returns one of the csv-files as a
    formatted dataframe.

        Parameters:
            path_to_zip (str): path to the zip-file containing the dataset
            csv_index (int): index (0-49) of the desired csv-file

        Returns:
            data_frame (Pandas DataFrame): dataframe converted from csv
    """

    # read df from csv within zip, parse timestamp as datetime and use as index
    time_00 = time.time()
    filename_prefix = os.path.basename(os.path.splitext(path_to_zip)[0])
    zip_file = zf.ZipFile(path_to_zip)
    csv_path = '{0:s}/{0:s}_{1:d}.csv'.format(filename_prefix, csv_index)

    time_01 = time.time()
    print(time_01 - time_00)

    data_frame = pd.read_csv(
        zip_file.open(csv_path),
        index_col='Unnamed: 0',
        parse_dates=True,
        )

    time_02 = time.time()
    print(time_02 - time_01)

    # add new columns with year, month, day, hour of timestamp
    data_frame['year'] = data_frame.index.year
    data_frame['month'] = data_frame.index.month
    data_frame['day'] = data_frame.index.day
    data_frame['hour'] = data_frame.index.hour
    data_frame['minute_of_hour'] = data_frame.index.minute
    data_frame['minute_of_day'] = data_frame['minute_of_hour'] + data_frame['hour'] * 60

    # return dataframe
    return data_frame


def extract_df_from_dir(path_to_dir, csv_index):

    """
    Takes a csvfile and returns a formatted dataframe.

        Parameters:
            path_to_dir (str): path to the directory containing the dataset

        Returns:
            data_frame (Pandas DataFrame): dataframe converted from csv
    """

    # read df from csv within dir, parse timestamp as datetime and use as index
    time_00 = time.time()
    filename_prefix = os.path.basename(os.path.splitext(path_to_dir)[0])
    csv_path = '{0:s}/{0:s}_{1:d}.csv'.format(filename_prefix, csv_index)

    time_01 = time.time()
    print(time_01 - time_00)

    data_frame = pd.read_csv(
        csv_path,
        index_col='Unnamed: 0',
        parse_dates=True,
        )

    time_02 = time.time()
    print(time_02 - time_01)

    # add new columns with year, month, day, hour of timestamp
    data_frame['year'] = data_frame.index.year
    data_frame['month'] = data_frame.index.month
    data_frame['day'] = data_frame.index.day
    data_frame['hour'] = data_frame.index.hour
    data_frame['minute_of_hour'] = data_frame.index.minute
    data_frame['minute_of_day'] = data_frame['minute_of_hour'] + data_frame['hour'] * 60

    # return dataframe
    return data_frame
