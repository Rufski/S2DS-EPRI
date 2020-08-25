from  cods_analysis import CODS_with_bootstrap
import sys

if __name__ == "__main__":

    synth_type = sys.argv[1]

    for i in range(0, 50):
        CODS_with_bootstrap(synth_type, index=i, realizations=512, verbose=True)
