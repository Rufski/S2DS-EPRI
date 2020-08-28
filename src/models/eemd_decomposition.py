from src.data.import_data import import_df_from_zip_pkl, import_df_info_from_zip
from src.data.make_dataset import downsample_dataframe
import numpy as np
from scipy import fftpack
from PyEMD import EMD, EEMD


def eemd_decomposition(signal, seed=True):
    """
    Decomposes signal into a set of oscillatory functions
    named Intrinsic Mode Functions (IMFs)

    """
    # Initialize EEMD class
    eemd   = EEMD(spline_kind="cubic", extrema_detection="parabol", trials=200, noise_width=0.01)
    if seed:
        eemd.noise_seed(1)
    eIMFs  = eemd.eemd(signal)
    _, res = eemd.get_imfs_and_residue()
    nIMFs  = eIMFs.shape[0]
    return eIMFs, res, nIMFs

def ft_imf(eIMFs, nIMFs):
    box_length  = eIMFs[0].size
    #sample_freq = fftpack.fftfreq(box_length, d=1)
    power_IMFs  = np.ones((nIMFs, box_length), dtype=complex)
    for n in range(nIMFs):
        power_IMFs[n] = fftpack.fft(eIMFs[n])
    return power_IMFs

# m denotes the number of examples
def gradientDescent(x, y, theta, alpha, m, numIterations=100000):
    """
    Batch gradient descent for linear regression
    """
    for i in range(0, numIterations):
        hypothesis = np.transpose(np.matmul(x, theta))
        #loss       = hypothesis - y
        #cost       = np.sum(loss**2)/(2*m)
        #print("Iteration %d | Cost: %f" % (i, cost))
        theta      = theta - alpha/m*np.transpose(np.matmul(hypothesis - y, x))
    return theta


def group_imfs_into_trend(sample_freq, power, box_length, eIMFs, nIMFs, max_extrema=2):
    """
    """
    emd   = EMD()
    trend = np.zeros(box_length)
    for n in range(nIMFs):
        max_pos, _, min_pos, _, _ = emd.find_extrema(sample_freq, np.abs(power[n]))
        if  max_pos.shape[0]<max_extrema and min_pos.shape[0]<max_extrema:
            trend = trend + eIMFs[n]
    # degradation factor
    eta_d = np.exp(trend)/np.exp(trend[0]) ##### THIS MIGHT INVESTIGATE FURTHER!!
    return (trend, eta_d)

def eemd_analysis(path_to_data, index=0, sampling_function=np.mean, seed=True, max_extrema=2, verbose=False):
    """
    """
    # Load time series
    df      = import_df_from_zip_pkl(path_to_data, index=index, verbose=verbose)
    df_info = import_df_info_from_zip(path_to_data, verbose=verbose)
    rd_true = df_info.Degradation_rate_linear.to_numpy()[index]
    # Preprocessed input signal
    df      = downsample_dataframe(df, night_method="basic", clip_method="universal",
                                   power_sampling_function=sampling_function)
    df["ln_power"] = np.log(df.Power.to_numpy())
    # Run EEMD
    eIMFs, res, nIMFs = eemd_decomposition(df.ln_power.to_numpy())
    # Fourier transfor
    power       = ft_imf(eIMFs, nIMFs)
    box_length  = eIMFs[0].size
    sample_freq = fftpack.fftfreq(box_length, d=1)
    # Estimate degradation factor
    trend, eta_d = group_imfs_into_trend(sample_freq, power, box_length, eIMFs, nIMFs,
                                         max_extrema=max_extrema)
    # Estimate degradation factor
    theta_init = np.array([[1], [0]]) # initial guess for linear regression parameters
    x          = np.concatenate((np.ones(len(trend)), np.arange(1, len(trend) + 1, 1)))
    x          = np.transpose(x.reshape(2, len(trend)))
    y          = np.array(eta_d, ndmin=2)
    theta      = gradientDescent(x, y, theta_init, 1e-7, len(trend), numIterations=1000)

    eta_d_1    = np.matmul(x, theta)
    time       = np.arange(1, len(eta_d_1) + 1, 1) / 365. # [years]
    rd_pred    = np.mean(np.divide(eta_d_1[:, 0] - 1, time))

    return ([eIMFs, res, nIMFs, power], [rd_pred, rd_true, eta_d_1, eta_d, trend])
