from src.data.import_data import import_df_from_zip_pkl
import rdtools
import pickle
import time


def CODS_with_bootstrap(synth_type, index=0, realizations=512, verbose=False):
    """

    """
    # Load datasets
    path_to_zip_pkl_pi = "../../data/raw/synthetic_" + synth_type + "_pi_daily.zip"
    try:
        df = import_df_from_zip_pkl(path_to_zip_pkl_pi, index=index, verbose=True, minofday=False)
    except:
        if verbose:
            print("No available synthetic dataset, the availabe types are 'basic', 'soil', 'soil_weather', 'weather'")

    start_time = time.time() # remove?

    # Initialize instance
    cods_instance = rdtools.soiling.cods_analysis(df.PI)
    # run algoritm
    cods_instance.run_bootstrap(realizations, verbose=verbose)

    end_time = time.time() # remove?
    print("--- %s min ---" %((end_time - start_time)/60.)) # remove?

    # save results
    _file = open("../../data/processed/cods_results_" + synth_type + "_" + str(index) + "_" + str(realizations) + ".pkl", "wb")
    pickle.dump(cods_instance , _file)

    return


def load_CODS_results(synth_type, index=0, realizations=512, verbose=False):
    """

    """
    # Load results
    try:
        _file         = open("../../data/processed/cods_results_" + synth_type + "_" + str(index) + "_" + str(realizations) + ".pkl", "rb")
        print (_file)
    except:
        if verbose:
            print("No available synthetic dataset, the availabe types are 'basic', 'soil', 'soil_weather', 'weather'")

    return cods_instance
