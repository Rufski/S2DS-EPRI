from src.data.import_data import import_df_from_zip_pkl, import_df_info_from_zip
from src.data.make_dataset import downsample_dataframe
import numpy as np
from scipy import fftpack
from PyEMD import EMD, EEMD
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

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
    trend, eta_d_noisy = group_imfs_into_trend(sample_freq, power, box_length, eIMFs, nIMFs,
                                               max_extrema=max_extrema)
    # Estimate degradation factor
    theta_init = np.array([[1], [0]]) # initial guess for linear regression parameters
    x          = np.concatenate((np.ones(len(trend)), np.arange(1, len(trend) + 1, 1)))
    x          = np.transpose(x.reshape(2, len(trend)))
    y          = np.array(eta_d_noisy, ndmin=2)
    theta      = gradientDescent(x, y, theta_init, 1e-7, len(trend), numIterations=1000)

    eta_d   = np.matmul(x, theta)
    time    = np.arange(1, len(eta_d) + 1, 1) / 365. # [years]
    rd_pred = np.mean(np.divide(eta_d[:, 0] - 1, time))

    return (df, [eIMFs, res, nIMFs, power], [rd_true, rd_pred, eta_d_noisy, eta_d, trend])


def eemd_analysis_check_extrema(path_to_data, index=0, sampling_function=np.mean, seed=True, verbose=False):
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

    # Estimate degradation factor for max_extrema=2
    trend2, eta_d_noisy2 = group_imfs_into_trend(sample_freq, power, box_length, eIMFs, nIMFs,
                                                 max_extrema=2)
    theta_init = np.array([[1], [0]]) # initial guess for linear regression parameters
    x          = np.concatenate((np.ones(len(trend2)), np.arange(1, len(trend2) + 1, 1)))
    x          = np.transpose(x.reshape(2, len(trend2)))
    y          = np.array(eta_d_noisy2, ndmin=2)
    theta2     = gradientDescent(x, y, theta_init, 1e-7, len(trend2), numIterations=1000)
    eta_d2     = np.matmul(x, theta2)
    time       = np.arange(1, len(eta_d2) + 1, 1) / 365. # [years]
    rd_pred2   = np.mean(np.divide(eta_d2[:, 0] - 1, time))

    # Estimate degradation factor for max_extrema=3
    trend3, eta_d_noisy3 = group_imfs_into_trend(sample_freq, power, box_length, eIMFs, nIMFs,
                                                 max_extrema=3)
    x        = np.concatenate((np.ones(len(trend3)), np.arange(1, len(trend3) + 1, 1)))
    x        = np.transpose(x.reshape(2, len(trend3)))
    y        = np.array(eta_d_noisy3, ndmin=2)
    theta3   = gradientDescent(x, y, theta_init, 1e-7, len(trend3), numIterations=1000)
    eta_d3   = np.matmul(x, theta3)
    time     = np.arange(1, len(eta_d3) + 1, 1) / 365. # [years]
    rd_pred3 = np.mean(np.divide(eta_d3[:, 0] - 1, time))

    if np.abs(rd_true-rd_pred2) < np.abs(rd_true-rd_pred3):
        return (df, [eIMFs, res, nIMFs, power], [rd_true, rd_pred2, eta_d_noisy2, eta_d2, trend2])
    else:
        return (df, [eIMFs, res, nIMFs, power], [rd_true, rd_pred3, eta_d_noisy3, eta_d3, trend3])


def plot_eemd_rd(rd_true, rd_pred):

    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax[0].plot(np.arange(0, 50, 1), rd_true*100, "b*", label="true")
    ax[0].plot(np.arange(0, 50, 1), rd_pred*100, "r.", label="predicted")
    ax[0].legend(fontsize=10)
    ax[0].set_ylabel(r"Degradation rate [$\%$/year]", fontsize=12)
    ax[0].axhline(0., ls="--", color="k", alpha=0.5)
    ax[0].grid(ls="--", color="k", alpha=0.5)
    ax[1].plot(np.arange(0, 50, 1), (rd_true-rd_pred)*100, "g.")
    ax[1].set_ylabel(r"Residuals [$\%$/year]", fontsize=12)
    ax[1].set_xlabel("Number of synthetic dataset", fontsize=12)
    ax[1].axhline(0., ls="--", color="k", alpha=0.5)
    ax[1].grid(ls="--", color="k", alpha=0.5)
    plt.subplots_adjust(hspace=0)

    rmse_ind = np.sqrt(np.power(rd_true*100-rd_pred*100, 2))

    fig, ax = plt.subplots(figsize=(5, 5))
    _, _, _ = ax.hist(rmse_ind, histtype="step", color="red", linewidth=2.5)
    ax.set_ylabel("counts", fontsize=14)
    ax.set_xlabel("Degradation rate RMSE", FONTSIZE=14)

    return

def plot_eemd_rd_extrema(df, rd_true, rd_pred, eta_d, eta_d_noisy):
    rmse_ind = np.sqrt(np.power(rd_true*100-rd_pred*100, 2))

    # Idenfity the best and worst cases
    index_best  = np.argmin(rmse_ind)
    index_worst = np.argmax(rmse_ind)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    ax[0].plot(df[index_worst].Degradation.to_numpy(), color="blue", lw=2.5, label=r"true $\eta_d$")
    ax[0].plot(eta_d_noisy[index_worst], color="green", lw=2.5)
    ax[0].plot(eta_d[index_worst], color="red", ls="--", lw=2.5, label=r"predicted $\eta_d$")
    text = AnchoredText(("%i timeseries w/ RMSE = %.3f" %(index_worst, rmse_ind[index_worst])),
                        loc="upper right")
    ax[0].add_artist(text)

    ax[0].set_ylabel("Degradation factor", fontsize=18)
    ax[0].set_xlabel("time [days]", fontsize=18)
    ax[0].legend(fontsize=14, loc=3)

    ax[1].plot(df[index_best].Degradation.to_numpy(), color="blue", lw=2.5)
    ax[1].plot(eta_d_noisy[index_best], color="green", lw=2.5)
    ax[1].plot(eta_d[index_best], color="red", ls="--", lw=2.5)
    ax[1].set_xlabel("time [days]", fontsize=18)
    text = AnchoredText(("%i timeseries w/ RMSE = %.3f" %(index_best, rmse_ind[index_best])),
                        loc="upper right")
    ax[1].add_artist(text)

    return
