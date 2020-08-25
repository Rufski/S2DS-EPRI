from cods_analysis import cods_with_bootstrap
import sys

if __name__ == "__main__":

    synth_type = sys.argv[1]

    for i in range(0, 25):
        print ("running CODS algorithm for %i time series" %i)
        cods_with_bootstrap(synth_type, index=i, realizations=16, clipping="basic", verbose=True)
