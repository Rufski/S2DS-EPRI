"""
module for data extraction and import functions
all functions of this module take files or directories as input and return dataframes
"""

import os
import zipfile as zf
import time
import pandas as pd


def import_df_from_zip(path_to_zip, csv_index, datetime=True, verbose=False):

    """
    Import a timeseries from a zipped datafile

    Extracts a csv file from the zip-archive and transforms it to a formatted dataframe.

        Args:
            path_to_zip (str): path to the zip-file containing the dataset
            csv_index (int): index (0-49) of the desired csv-file
            datetime (bool, optional): parse time-string to datetime, defaults to True
            verbose (bool, optional): print output if true, defaults to False

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
        )

    if datetime is True:
        data_frame.index = pd.to_datetime(data_frame.index)

    # print how much time read_csv needs
    time_01 = time.time()
    if verbose is True:
        print("time for importing dataframe: {:.2f} seconds".format(time_01 - time_00))

    # add new column for minute of day
    if datetime is True:
        data_frame['minute_of_hour'] = data_frame.index.minute + data_frame.index.hour * 60

    # return dataframe
    return data_frame


def import_df_from_dir(path_to_dir, csv_index, datetime=True, verbose=False):

    """
    Import a timeseries from a data directory

    Takes csv file from the destination directory and returns a formatted dataframe.

        Args:
            path_to_dir (str): path to the directory containing the csv-files
            csv_index (int): index (0-49) of the desired csv-file
            datetime (bool, optional): parse time-string to datetime, defaults to True
            verbose (bool, optional): print output if true, defaults to False

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
        )

    if datetime is True:
        data_frame.index = pd.to_datetime(data_frame.index)

    # print how much time read_csv needs
    time_01 = time.time()
    if verbose is True:
        print("time for importing dataframe: {:.2f} seconds".format(time_01 - time_00))

    # add new column for minute of day
    if datetime is True:
        data_frame['minute_of_hour'] = data_frame.index.minute + data_frame.index.hour * 60

    # return dataframe
    return data_frame

def import_df_from_pickle_zip(path_to_zip, csv_index, verbose=False):

    """
    Import a timeseries from a zipped pickled dataframe

    Extracts a dataframe file from the pickle (compressed using gzip)
    which is saved within a zipped folder

        Args:
            path_to_zip (str): path to the zip-file containing the pickled dataframes
            csv_index (int): index (0-49) of the desired csv-file
            verbose (bool, optional): print output if true, defaults to False

        Returns:
            data_frame (Pandas DataFrame): unpickled dataframe
    """

    # unpickle df from within within zip
    time_00 = time.time()
    zip_file = zf.ZipFile(path_to_zip)
    filename_prefix = os.path.basename(os.path.splitext(path_to_zip)[0])
    csv_path = '{0:s}/{0:s}_{1:d}.csv'.format(filename_prefix, csv_index)
    data_frame = pd.read_pickle(zip_file.open(csv_path), compression='gzip')
    time_01 = time.time()

    # print how much time read_pickle needs
    if verbose is True:
        print("time for importing dataframe: {:.2f} seconds".format(time_01 - time_00))

    # return dataframe
    return data_frame
