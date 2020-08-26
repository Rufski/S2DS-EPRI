import numpy as np
import pandas as pd
import tensorflow as tf

def load_and_prepare_PI_data(data_dir_path,
                             nr_files,
                             nr_years,
                             train_valid_test_split = (0.70, 0.15, 0.15),
                             clipping = 'basic',
                             verbose=False):
    """
    Loads normalised daily dataset and splits it into train, valid, and testing sets.

    The function removes 29th of February and expects collumns
    PI_clipping_basic, PI_clipping_flexible, and PI_clipping_universal.

        Args:
            data_dir_path (str): path to the zip of pickles of the datasets
            nr_files (int): number of files in the zip
            nr_years (int): number of years per signal
            train_valid_test_split (tuple of lenght 3, optional): proportions
                of train, valid, and test sets   
            clipping (str, optional): clipping used to to compute PI, must be
                either 'basic', 'flexible', or 'universal'

        Returns:
            tuple: first three elements are the PI signals of train, valid, and
                test sets, and the later three elements are the degradation
                rates of the corresponding signals
    """
    
    if np.sum(train_valid_test_split) != 1.:
        print("Error")
        return
    
    column_name = 'PI_clipping_' + 'basic'
    
    X = np.empty((0, 365*nr_years), float)
    y = np.empty((0, 365*nr_years), float)
        
    for i in range(nr_files):
        if verbose:
            print("Loading file #"+str(i)+" from dataset "+dataset.split("/")[-1])
        
        df = import_df_from_zip_pkl(data_dir_path, index=i, verbose=verbose, minofday=False)
        # Remove 29th February from time series
        df = df[(df.index.month != 2) | (df.index.day != 29)]
        df[column_name] = df[column_name].ffill()

        y = np.vstack([
            y,
            np.array(df[(df.index.hour == 0) & (df.index.minute == 0)].Degradation)])
        X = np.vstack([X, np.array(df[column_name])])

    if verbose:
        print("All processed!")
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_valid_test_split[2], random_state=42)
    valid_ratio = train_valid_test_split[1] / (1 - train_valid_test_split[2])
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_ratio, random_state=42) 
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test
