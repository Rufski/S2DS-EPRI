"""
module for data extraction and import functions
all functions of this module take files or directories as input and return dataframes
"""

import os
import zipfile as zf
import time
import pandas as pd

def import_df_from_zip(path_to_zip, csv_index, verbose=False):

    """
    Import a timeseries from a zipped datafile

    Extracts a csv file from the zip-archive and transforms it to a formatted dataframe.

        Args:
            path_to_zip (str): path to the zip-file containing the dataset
            csv_index (int): index (0-49) of the desired csv-file
            verbose (bool): optional, print output if true

        Returns:
            data_frame (Pandas DataFrame): dataframe converted from csv
    """

    # read df from csv within zip, parse timestamp as datetime and use as index
    filename_prefix = os.path.basename(os.path.splitext(path_to_zip)[0])
    zip_file = zf.ZipFile(path_to_zip)
    csv_path = '{0:s}/{0:s}_{1:d}.csv'.format(filename_prefix, csv_index)

    time_00 = time.time()
    data_frame = pd.read_csv(
        zip_file.open(csv_path),
        index_col='Unnamed: 0',
        parse_dates=True,
        )

    # print how much time read_csv needs
    time_01 = time.time()
    if verbose is True:
        print("time for importing dataframe: {:.2f} seconds".format(time_01 - time_00))

    # add new columns with year, month, day, hour of timestamp
    data_frame['year'] = data_frame.index.year
    data_frame['month'] = data_frame.index.month
    data_frame['day'] = data_frame.index.day
    data_frame['hour'] = data_frame.index.hour
    data_frame['minute_of_hour'] = data_frame.index.minute
    data_frame['minute_of_day'] = data_frame['minute_of_hour'] + data_frame['hour'] * 60

    # return dataframe
    return data_frame


def import_df_from_dir(path_to_dir, csv_index, verbose=False):

    """
    Import a timeseries from a data directory
    
    Takes csv file from the destination directory and returns a formatted dataframe.

        Args:
            path_to_dir (str): path to the directory containing the csv-files
            csv_index (int): index (0-49) of the desired csv-file
            verbose (bool): optional, print output if true

        Returns:
            data_frame (Pandas DataFrame): dataframe converted from csv
    """

    # read df from csv within dir
    # this assumes the folder structure is according to our conventions
    # also parses timestamp as datetime and use as index
    filename_prefix = os.path.basename(os.path.normpath(path_to_dir))

    csv_path = '{0:s}/{1:s}_{2:d}.csv'.format(path_to_dir, filename_prefix, csv_index)

    time_00 = time.time()
    data_frame = pd.read_csv(
        csv_path,
        index_col='Unnamed: 0',
        parse_dates=True,
        )

    # print how much time read_csv needs
    time_01 = time.time()
    if verbose is True:
        print("time for importing dataframe: {:.2f} seconds".format(time_01 - time_00))

    # add new columns with year, month, day, hour of timestamp
    data_frame['year'] = data_frame.index.year
    data_frame['month'] = data_frame.index.month
    data_frame['day'] = data_frame.index.day
    data_frame['hour'] = data_frame.index.hour
    data_frame['minute_of_hour'] = data_frame.index.minute
    data_frame['minute_of_day'] = data_frame['minute_of_hour'] + data_frame['hour'] * 60

    # return dataframe
    return data_frame
