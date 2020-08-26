from src.data.import_data import import_df_from_zip_pkl
import rdtools
import pickle
import time
import numpy as np

def cods_with_bootstrap_extra(synthetic_type, index=0, realizations=512, clipping="basic", extra=False, outlier_threshold=0., verbose=False):
    
    """
    
    """

    # Load datasets
    path_to_zip_pkl_pi = f'../../data/raw/new/synthetic_{synthetic_type:s}_pi_daily_extra.zip'
    try:
        df = import_df_from_zip_pkl(path_to_zip_pkl_pi, index=index, verbose=True, 
                                    minofday=False)
    except:
        if verbose:
            print("No available synthetic dataset, the availabe types are 'basic', 'soil', 'soil_weather', 'weather'")
    
    start_time = time.time() # remove?
    
    # Initialize instance
    if clipping=="basic":
        cods_n = 1
        print(df.PI_clipping_basic.isna().sum())
        df[df.PI_clipping_basic<outlier_threshold] = np.nan
        print(df.PI_clipping_basic.isna().sum())
        cods_instance = rdtools.soiling.cods_analysis(df.PI_clipping_basic)
    elif clipping=="flexible":
        cods_n = 2
        df[df.PI_clipping_flexible<outlier_threshold] = np.nan
        cods_instance = rdtools.soiling.cods_analysis(df.PI_clipping_flexible)
    elif clipping=="universal":
        cods_n = 3
        df[df.PI_clipping_universal<outlier_threshold] = np.nan
        cods_instance = rdtools.soiling.cods_analysis(df.PI_clipping_universal)
    else:
        if verbose==True:
            print("Function for removing clipping not implemented!")

    # run algorithm
    cods_instance.run_bootstrap(realizations, verbose=verbose)
    
    end_time = time.time() # remove?
    print("--- %s min ---" %((end_time - start_time)/60.)) # remove?
    
    # save results
    type_name = (synthetic_type, synthetic_type + '_extra')[extra == True] 
    _file = open(f'../../data/processed/new/cods_results_{synthetic_type:s}_extra_i{str(index).zfill(3):s}_r{str(realizations).zfill(3):s}_c{cods_n:d}.pkl', "wb")
    pickle.dump(cods_instance , _file)
    
    return None

def cods_with_bootstrap_uniform(synthetic_type, index=0, realizations=512, clipping="basic", extra=False, outlier_threshold=0., verbose=False):
    
    """
    
    """

    # Load datasets
    path_to_zip_pkl_pi = f'../../data/raw/new/synthetic_{synthetic_type:s}_pi_daily_uniform.zip'
    try:
        df = import_df_from_zip_pkl(path_to_zip_pkl_pi, index=index, verbose=True, 
                                    minofday=False)
    except:
        if verbose:
            print("No available synthetic dataset, the availabe types are 'basic', 'soil', 'soil_weather', 'weather'")
    
    start_time = time.time() # remove?
    
    # Initialize instance
    if clipping=="basic":
        cods_n = 1
        print(df.PI_clipping_basic.isna().sum())
        df[df.PI_clipping_basic<outlier_threshold] = np.nan
        print(df.PI_clipping_basic.isna().sum())
        cods_instance = rdtools.soiling.cods_analysis(df.PI_clipping_basic)
    elif clipping=="flexible":
        cods_n = 2
        df[df.PI_clipping_flexible<outlier_threshold] = np.nan
        cods_instance = rdtools.soiling.cods_analysis(df.PI_clipping_flexible)
    elif clipping=="universal":
        cods_n = 3
        df[df.PI_clipping_universal<outlier_threshold] = np.nan
        cods_instance = rdtools.soiling.cods_analysis(df.PI_clipping_universal)
    else:
        if verbose==True:
            print("Function for removing clipping not implemented!")

    # run algorithm
    cods_instance.run_bootstrap(realizations, verbose=verbose)
    
    end_time = time.time() # remove?
    print("--- %s min ---" %((end_time - start_time)/60.)) # remove?
    
    # save results
    type_name = (synthetic_type, synthetic_type + '_extra')[extra == True] 
    _file = open(f'../../data/processed/new/cods_results_{synthetic_type:s}_uniform_i{str(index).zfill(3):s}_r{str(realizations).zfill(3):s}_c{cods_n:d}.pkl', "wb")
    pickle.dump(cods_instance , _file)
    
    return None

def cods_with_bootstrap(synthetic_type, index=0, realizations=512, clipping="basic", extra=False, outlier_threshold=0., verbose=False):
    
    """
    
    """

    # Load datasets
    path_to_zip_pkl_pi = f'../../data/raw/new/synthetic_{synthetic_type:s}_pi_daily.zip'
    try:
        df = import_df_from_zip_pkl(path_to_zip_pkl_pi, index=index, verbose=True, 
                                    minofday=False)
    except:
        if verbose:
            print("No available synthetic dataset, the availabe types are 'basic', 'soil', 'soil_weather', 'weather'")
    
    start_time = time.time() # remove?
    
    # Initialize instance
    if clipping=="basic":
        cods_n = 1
        print(df.PI_clipping_basic.isna().sum())
        df[df.PI_clipping_basic<outlier_threshold] = np.nan
        print(df.PI_clipping_basic.isna().sum())
        cods_instance = rdtools.soiling.cods_analysis(df.PI_clipping_basic)
    elif clipping=="flexible":
        cods_n = 2
        df[df.PI_clipping_flexible<outlier_threshold] = np.nan
        cods_instance = rdtools.soiling.cods_analysis(df.PI_clipping_flexible)
    elif clipping=="universal":
        cods_n = 3
        df[df.PI_clipping_universal<outlier_threshold] = np.nan
        cods_instance = rdtools.soiling.cods_analysis(df.PI_clipping_universal)
    else:
        if verbose==True:
            print("Function for removing clipping not implemented!")

    # run algorithm
    cods_instance.run_bootstrap(realizations, verbose=verbose)
    
    end_time = time.time() # remove?
    print("--- %s min ---" %((end_time - start_time)/60.)) # remove?
    
    # save results
    type_name = (synthetic_type, synthetic_type + '_extra')[extra == True] 
    _file = open(f'../../data/processed/new/cods_results_{synthetic_type:s}_i{str(index).zfill(3):s}_r{str(realizations).zfill(3):s}_c{cods_n:d}.pkl', "wb")
    pickle.dump(cods_instance , _file)
    
    return None

def load_cods_results(synth_type, index=0, clipping="basic", verbose=False):
    """

    """
    path = ("../data/processed/cods_results_" + synth_type
             + "_r" + str(realizations) + "_c" + clipping + "_o" + outlier + ".zip")

    cods_instance = import_cods_instance_from_zip_pkl(path, index=index, verbose=verbose)

    return cods_instance
