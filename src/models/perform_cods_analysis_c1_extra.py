from cods_analysis import cods_with_bootstrap_extra
import sys

if __name__ == "__main__":

    synth_type = sys.argv[1]

    for i in range(200):
        print ("running CODS algorithm for %i time series" %i)
        cods_with_bootstrap_extra(synth_type, index=i, realizations=512, clipping="basic", extra=False, verbose=True, outlier_threshold=0.0)
