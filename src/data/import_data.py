"""
module for data import and export functions
"""

import os
import time
import pickle
import zipfile as zf
import pandas as pd
import pickle

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

    list_filename.sort()
    path_to_csv = list_filename[index]

    if verbose is True:
        print(f'\nimporting {path_to_csv:s}\n')

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
    list_files_csv.sort()
    path_to_csv = '{0:s}/{1:s}'.format(path_to_dir, list_files_csv[index])

    if verbose is True:
        print(f'\nimporting {path_to_csv:s}\n')

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


def import_df_from_zip_pkl(path_to_zip, index=0, verbose=False, minofday=True):

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
    list_filename = [file_in_zip.filename for file_in_zip
                     in list_fileinfos if '.pkl' in file_in_zip.filename]

    list_filename.sort()
    path_to_pkl = list_filename[index]

    if verbose is True:
        print(f'\nimporting {path_to_pkl:s}\n')

    data_frame = pd.read_pickle(zip_file.open(path_to_pkl), compression='gzip')
    time_01 = time.time()

    # add column for minute of day
    if minofday is True:
        data_frame['minute_of_day'] = data_frame.index.minute + \
            data_frame.index.hour * 60

    # print how much time read_pickle needs
    if verbose is True:
        print("time for importing dataframe: "
              "{:.2f} seconds".format(time_01 - time_00))

    # return dataframe
    return data_frame


def import_df_info_from_zip(path_to_zip, verbose=False):

    """
    Import the dataset info csv-file from a zipped dataset

    Imports the dataset info as a pandas dataframe from the csv-file from
    within a zipped dataset (without unzipping)

        Args:
            path_to_zip (str): path to the zip-file containing the dataset
            verbose (bool, optional): print output if true, defaults to False

        Returns:
            data_frame (Pandas DataFrame): dataframe containing dataset info
    """

    zip_file = zf.ZipFile(path_to_zip)
    list_fileinfos = zip_file.filelist
    list_filename = [file_in_zip.filename for file_in_zip
                     in list_fileinfos if '.csv' in file_in_zip.filename]

    if not list_filename:
        print("\nno dataset info file found!\n")

    if verbose is True:
        print("Found the following dataset info-files ", list_filename)

    path_to_csv = list_filename[0]

    if verbose is True:
        print(f'\nimporting {path_to_csv:s}\n')

    data_frame = pd.read_csv(zip_file.open(path_to_csv))

    # return dataframe
    return data_frame


def import_cods_instance_from_zip_pkl(path_to_zip, index=0, verbose=False):

    """
    Import a pickled CODS results instance from a zip archive

    Extracts a cods instance from the pickle (compressed using gzip)
    which is saved within a zipped folder

        Args:
            path_to_zip (str): path to the zip-file containing the pickled
                cods instance
            index (int, optional): index (0-49) of the desired file, defaults
            to 0, ie, the first timeseries in the dataset
            verbose (bool, optional): print output if true, defaults to False

        Returns:
            cods_instance (Python Class Instance): unpickled cods_instance
    """

    # unpickle instance from within within zip
    time_00 = time.time()
    zip_file = zf.ZipFile(path_to_zip)
    list_fileinfos = zip_file.filelist
    list_filename = [file_in_zip.filename for file_in_zip
                     in list_fileinfos if '.pkl' in file_in_zip.filename]
    path_to_pkl = list_filename[index]

    list_filename.sort()
    path_to_pkl = list_filename[index]

    if verbose is True:
        print(f'\nimporting {path_to_pkl:s}\n')

    infile = zip_file.open(path_to_pkl)
    cods_instance = pickle.load(infile)
    infile.close()
    time_01 = time.time()

    # print how much time read_pickle needs
    if verbose is True:
        print("time for importing dataframe: "
              "{:.2f} seconds".format(time_01 - time_00))

    # return dataframe
    return cods_instance
