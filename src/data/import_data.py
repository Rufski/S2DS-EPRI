"""
module for data import and export functions
"""

import os
import zipfile as zf
import time
import pandas as pd


def import_df_from_zip_csv(path_to_zip, index=0, datetime=True, verbose=False):

    """
    Import a timeseries from a zipped datafile

    Extracts a csv file from the zip-archive and transforms it to a formatted
    dataframe.

        Args:
            path_to_zip (str): path to the zip-file containing the dataset
            index (int, optional): index (0-49) of the desired timeseries,
                defaults to 0, ie, the first timeseries in the dataset
            datetime (bool, optional): parse time-string to datetime, defaults
                to True
            verbose (bool, optional): print output if true, defaults to False

        Returns:
            data_frame (Pandas DataFrame): dataframe converted from csv
    """

    # read df from csv within zip, parse timestamp as datetime and use as index
    zip_file = zf.ZipFile(path_to_zip)
    list_fileinfo = zip_file.filelist
    list_filename = [file_in_zip.filename for file_in_zip
                     in list_fileinfo if '.csv' in file_in_zip.filename]
    path_to_csv = list_filename[index]

    time_00 = time.time()
    data_frame = pd.read_csv(
        zip_file.open(path_to_csv),
        index_col='Unnamed: 0',
        )

    if datetime is True:
        data_frame.index = pd.to_datetime(data_frame.index)

    # print how much time read_csv needs
    time_01 = time.time()
    if verbose is True:
        print("time for importing dataframe: "
              "{:.2f} seconds".format(time_01 - time_00))

    # add new column for minute of day
    if datetime is True:
        data_frame['minute_of_day'] = data_frame.index.minute + \
            data_frame.index.hour * 60

    # return dataframe
    return data_frame


def import_df_from_dir_csv(path_to_dir, index=0, datetime=True, verbose=False):

    """
    Import a timeseries from a data directory

    Takes csv file from the destination directory and returns a formatted
    dataframe.

        Args:
            path_to_dir (str): path to the directory containing the csv-files
            index (int, optional): index (0-49) of the desired timeseries,
                defaults to 0, ie, the first timeseries in the dataset
            datetime (bool, optional): parse time-string to datetime, defaults
                to True
            verbose (bool, optional): print output if true, defaults to False

        Returns:
            data_frame (Pandas DataFrame): dataframe converted from csv
    """

    # read df from csv within dir
    # this assumes the folder structure is according to our conventions
    # also parses timestamp as datetime and use as index

    list_files_all = os.listdir(path_to_dir)
    list_files_csv = [file_in_dir for file_in_dir
                      in list_files_all if 'csv' in file_in_dir]
    path_to_csv = '{0:s}/{1:s}'.format(path_to_dir, list_files_csv[index])

    time_00 = time.time()
    data_frame = pd.read_csv(
        path_to_csv,
        index_col='Unnamed: 0',
        )

    if datetime is True:
        data_frame.index = pd.to_datetime(data_frame.index)

    # print how much time read_csv needs
    time_01 = time.time()
    if verbose is True:
        print("time for importing dataframe: "
              "{:.2f} seconds".format(time_01 - time_00))

    # add new column for minute of day
    if datetime is True:
        data_frame['minute_of_day'] = data_frame.index.minute + \
            data_frame.index.hour * 60

    # return dataframe
    return data_frame


def import_df_from_zip_pkl(path_to_zip, index=0, verbose=False):

    """
    Import a timeseries from a zipped pickled dataframe

    Extracts a dataframe file from the pickle (compressed using gzip)
    which is saved within a zipped folder

        Args:
            path_to_zip (str): path to the zip-file containing the pickled
                dataframes
            index (int, optional): index (0-49) of the desired timeseries,
                defaults to 0, ie, the first timeseries in the dataset
            verbose (bool, optional): print output if true, defaults to False

        Returns:
            data_frame (Pandas DataFrame): unpickled dataframe
    """

    # unpickle df from within within zip
    time_00 = time.time()
    zip_file = zf.ZipFile(path_to_zip)
    list_fileinfos = zip_file.filelist
    list_filenames = [file_in_zip.filename for file_in_zip
                      in list_fileinfos if '.pkl' in file_in_zip.filename]
    path_to_pkl = list_filenames[index]
    data_frame = pd.read_pickle(zip_file.open(path_to_pkl), compression='gzip')
    time_01 = time.time()

    data_frame['minute_of_day'] = data_frame.index.minute + \
        data_frame.index.hour * 60

    # print how much time read_pickle needs
    if verbose is True:
        print("time for importing dataframe: "
              "{:.2f} seconds".format(time_01 - time_00))

    # return dataframe
    return data_frame
