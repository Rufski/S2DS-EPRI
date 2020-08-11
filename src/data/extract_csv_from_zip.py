import pandas as pd
import zipfile as zf
import os


def extract_csv_from_zip(path_to_zip, csv_index):

    """
    function to extract csv data from zip and return dataframe
    """

    # read df from csv within zip, parse timestamp as datetime and use as index
    filename_prefix = os.path.basename(os.path.splitext(path_to_zip)[0])
    zip_file = zf.ZipFile(path_to_zip)
    csv_path = '{0:s}/{0:s}_{1:d}.csv'.format(filename_prefix, csv_index)
    
    data_frame = pd.read_csv(
            zip_file.open(csv_path),
            index_col='Unnamed: 0',
            parse_dates=True,
            )

    # add new columns with year, month, day, hour of timestamp
    data_frame['year'] = pd.DatetimeIndex(data_frame.index).year
    data_frame['month'] = pd.DatetimeIndex(data_frame.index).month
    data_frame['day'] = pd.DatetimeIndex(data_frame.index).day
    data_frame['hour'] = pd.DatetimeIndex(data_frame.index).hour
    data_frame['minute_of_hour'] = pd.DatetimeIndex(data_frame.index).minute
    data_frame['minute_of_day'] = data_frame['minute_of_hour'] + data_frame['hour'] * 60

    # return dataframe
    return data_frame
