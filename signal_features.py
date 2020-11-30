'''
This file houses all functions related to generating features from sensor signals.
'''
import numpy as np
import pandas as pd
from numba import jit
import time
from scipy import stats
import tsfresh as tsf
from sklearn import linear_model
from statsmodels.tsa.stattools import acf
from pyunicorn.timeseries.recurrence_network import RecurrenceNetwork

def timeit(method):
    '''
    Function to measure execution time.

    :param method: desired method to time
    :return: function to time method
    '''

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print "{}: {} sec".format(method.__name__, te-ts)
        return result

    return timed


@jit()
def signal_range(signal_df, channels):
    '''
    Calculate range of sensor signals.

    :param signal_df: dataframe housing desired sensor signals
    :param channels: channels of signal to measure range
    :return: dataframe housing calculated range for each signal channel
    '''
    range_df = pd.DataFrame()

    for channel in channels:
        range_df[channel + '_range'] = [signal_df[channel].max(skipna=True) - signal_df[channel].min(skipna=True)]

    return range_df


@jit()
def signal_iqr(signal_df, channels):
    #TODO: Update documentation
    '''

    :param signal_df: dataframe housing desired sensor signals
    :param channels: channels of signal to measure IQR
    :return:
    '''
    iqr_df = pd.DataFrame()

    for channel in channels:
        iqr_df[channel + '_iqr'] = [signal_df[channel].quantile(q=0.75) - signal_df[channel].quantile(q=0.25)]

    return iqr_df


@jit()
def signal_rms(signal_df, channels):
    '''
    Calculate root mean square of sensor signals.

    :param signal_df: dataframe housing desired sensor signals
    :param channels: channels of signal to measure RMS
    :return: dataframe housing calculated RMS for each signal channel
    '''
    rms_df = pd.DataFrame()

    for channel in channels:
        rms_df[channel + '_rms'] = [np.std(signal_df[channel] - signal_df[channel].mean())]

    return rms_df


@jit()
def signal_mean(signal_df, channels):
    '''
    Compute mean of sensor signals.

    :param signal_df: dataframe housing desired sensor signals
    :param channels: channels of signal to measure mean
    :return: dataframe housing calculated mean for each signal channel
    '''
    mean_df = pd.DataFrame()

    for channel in channels:
        mean_df[channel + '_mean'] = [signal_df[channel].mean()]

    return mean_df


@jit()
def signal_skewness(signal_df, channels):
    '''
    Compute skewness of sensor signals.

    :param signal_df: dataframe housing desired sensor signals
    :param channels: channels of signal to measure skewness
    :return: dataframe housing calculated skewness for each signal channel
    '''
    skewness_df = signal_df[channels].skew()
    skewness_df = pd.DataFrame(skewness_df.add_suffix('_skewness'))

    return skewness_df.T


@jit()
def signal_kurtosis(signal_df, channels):
    '''
    Compute kurtosis of sensor signals.

    :param signal_df: dataframe housing desired sensor signals
    :param channels: channels of signal to measure kurtosis
    :return: dataframe housing calculated kurtosis for each signal channel
    '''
    kurtosis_df = signal_df[channels].kurtosis()
    kurtosis_df = pd.DataFrame(kurtosis_df.add_suffix('_kurtosis'))

    return kurtosis_df.T


@jit()
def range_ratio(signal_df, channels):
    '''
    Compute range ratio of sensor signals.

    :param signal_df: dataframe housing desired sensor signals
    :param channels: channels of signal to measure range ratio
    :return: dataframe housing calculated range ratio for each signal channel
    '''
    range_ratio_df = pd.DataFrame()
    for channel in channels:
        rangeX = signal_df[channel[0]].max() - signal_df[channel[0]].min()
        rangeY = signal_df[channel[1]].max() - signal_df[channel[1]].min()

        range_ratio_df[channel[0] + '_' + channel[1] + '_range_ratio'] = [rangeX / rangeY]

    return range_ratio_df


@jit()
def mean_cross_rate(signal_df, channels):
    '''
    Compute mean cross rate of sensor signals.

    :param signal_df: dataframe housing desired sensor signals
    :param channels: channels of signal to measure mean cross rate
    :return: dataframe housing calculated mean cross rate for each signal channel
    '''
    mean_cross_rate_df = pd.DataFrame()
    signal_df_mean = signal_df[channels] - signal_df[channels].mean()

    for channel in channels:
        MCR = 0

        for i in range(len(signal_df_mean) - 1):
            if np.sign(signal_df_mean.loc[i, channel]) != np.sign(signal_df_mean.loc[i + 1, channel]):
                MCR += 1

        MCR = float(MCR) / len(signal_df_mean)

        mean_cross_rate_df[channel + '_mean_cross_rate'] = [MCR]

    return mean_cross_rate_df


def histogram(signal_x):
    #TODO: Change to dataframe
    '''
    Calculate histogram of sensor signal.

    :param signal_x: 1-D numpy array of sensor signal
    :return: Histogram bin values, descriptor
    '''
    descriptor = np.zeros(3)

    ncell = np.ceil(np.sqrt(len(signal_x)))

    max_val = np.nanmax(signal_x.values)
    min_val = np.nanmin(signal_x.values)

    delta = (max_val - min_val) / (len(signal_x) - 1)

    descriptor[0] = min_val - delta / 2
    descriptor[1] = max_val + delta / 2
    descriptor[2] = ncell

    h = np.histogram(signal_x, ncell.astype(int), range=(min_val, max_val))

    return h[0], descriptor

def signal_entropy(signal_df, channels):
    '''
    Calculate signal entropy of sensor signals.

    :param signal_df: dataframe housing desired sensor signals
    :param channels: channels of signal to measure signal entropy
    :return: dataframe housing calculated signal entropy for each signal channel
    '''
    signal_entropy_df = pd.DataFrame()

    for channel in channels:
        data_norm = signal_df[channel]/np.std(signal_df[channel])
        h, d = histogram(data_norm)

        lowerbound = d[0]
        upperbound = d[1]
        ncell = int(d[2])

        estimate = 0
        sigma = 0
        count = 0

        for n in range(ncell):
            if h[n] != 0:
                logf = np.log(h[n])
            else:
                logf = 0
            count = count + h[n]
            estimate = estimate - h[n] * logf
            sigma = sigma + h[n] * logf ** 2

        nbias = -(float(ncell) - 1) / (2 * count)

        estimate = estimate / count
        estimate = estimate + np.log(count) + np.log((upperbound - lowerbound) / ncell) - nbias

        # Scale the entropy estimate to stretch the range
        estimate = np.exp(estimate ** 2) - np.exp(0) - 1

        signal_entropy_df[channel + '_signal_entropy'] = [estimate]

    return signal_entropy_df

def sample_entropy(signal_df, channels):
    '''
    Calculate sample entropy of sensor signals.

    :param signal_df: dataframe housing desired sensor signals
    :param channels: channels of signal to measure sample entropy
    :return: dataframe of calculated sample entropy for each signal channel
    '''
    from pyentrp import entropy as ent
    sample_entropy_df = pd.DataFrame()

    for channel in channels:
        current_sample_ent = ent.sample_entropy(signal_df[channel], 4, 1)
        sample_entropy_df[channel + '_sample_entropy_m1'] = [current_sample_ent[0]]
        sample_entropy_df[channel + '_sample_entropy_m2'] = [current_sample_ent[1]]
        sample_entropy_df[channel + '_sample_entropy_m3'] = [current_sample_ent[2]]
        sample_entropy_df[channel + '_sample_entropy_m4'] = [current_sample_ent[3]]

    return sample_entropy_df


# Permutation entropy
def permutation_entropy(signal_df, channels):
    '''
    Calculate permutation entropy of sensor signals.

    :param signal_df: dataframe housing desired sensor signals
    :param channels: channels of signal to measure permutation entropy
    :return: dataframe of calculated permutation entropy for each signal channel
    '''
    from pyentrp import entropy as ent
    permutation_entropy_df = pd.DataFrame()

    for channel in channels:
        current_perm_ent = ent.permutation_entropy(signal_df[channel], 4, 1)
        permutation_entropy_df[channel + '_permutation_entropy'] = [current_perm_ent]

    return permutation_entropy_df


@jit()
def correlation_coefficient(signal_df, channels):
    '''
    Calculate correlation coefficient of sensor signals.

    :param signal_df: dataframe housing desired sensor signals
    :param channels: channels of signal to measure correlation coefficient
    :return: dataframe of calculated correlation coefficient for each signal channel
    '''
    corr_coef_df = pd.DataFrame()
    C = signal_df.corr()

    for channel in channels:
        corr_coef_df[channel[0] + '_' + channel[1] + '_corr_coef'] = [C[channel[0]][channel[1]]]

    return corr_coef_df


# @timeit
@jit()
def dominant_frequency(signal_df, sampling_rate, cutoff, channels):
    '''
    Calculate dominant frequency of sensor signals.

    :param signal_df: dataframe housing desired sensor signals
    :param sampling_rate: sampling rate of sensor signal
    :param cutoff: desired cutoff for filter
    :param channels: channels of signal to measure dominant frequency
    :return: dataframe of calculated dominant frequency for each signal channel
    '''
    # TODO: Refactor this implementation
    dominant_freq_df = pd.DataFrame()

    for channel in channels:
        signal_x = signal_df[channel]

        padfactor = 1
        dim = signal_x.shape
        nfft = 2 ** ((dim[0] * padfactor).bit_length())

        freq_hat = np.fft.fftfreq(nfft) * sampling_rate
        freq = freq_hat[0:nfft / 2]

        idx1 = freq <= cutoff
        idx_cutoff = np.argwhere(idx1)
        freq = freq[idx_cutoff]

        sp_hat = np.fft.fft(signal_x, nfft)
        sp = sp_hat[0:nfft / 2] * np.conjugate(sp_hat[0:nfft / 2])
        sp = sp[idx_cutoff]
        sp_norm = sp / sum(sp)

        max_freq = freq[sp_norm.argmax()][0]
        max_freq_val = sp_norm.max().real

        idx2 = (freq > max_freq - 0.5) * (freq < max_freq + 0.5)
        idx_freq_range = np.where(idx2)[0]
        dom_freq_ratio = sp_norm[idx_freq_range].real.sum()

        # Calculate spectral flatness
        spectral_flatness = 10.0*np.log10(stats.mstats.gmean(sp_norm)/np.mean(sp_norm))

        # Estimate spectral entropy
        spectral_entropy_estimate = 0
        for isess in range(len(sp_norm)):
            if sp_norm[isess] != 0:
                logps = np.log2(sp_norm[isess])
            else:
                logps = 0
            spectral_entropy_estimate = spectral_entropy_estimate - logps * sp_norm[isess]

        spectral_entropy_estimate = spectral_entropy_estimate / np.log2(len(sp_norm))
        # spectral_entropy_estimate = (spectral_entropy_estimate - 0.5) / (1.5 - spectral_entropy_estimate)

        dominant_freq_df[channel + '_dom_freq_value'] = [max_freq]
        dominant_freq_df[channel + '_dom_freq_magnitude'] = [max_freq_val]
        dominant_freq_df[channel + '_dom_freq_ratio'] = [dom_freq_ratio]
        dominant_freq_df[channel + '_spectral_flatness'] = [spectral_flatness[0].real]
        dominant_freq_df[channel + '_spectral_entropy'] = [spectral_entropy_estimate[0].real]

    return dominant_freq_df


@jit()
def range_count_percentage(signal_df, channels, min_value=-1, max_value=1):
    '''
    Calculate range count percentage of sensor signals.

    :param signal_df: dataframe housing desired sensor signals
    :param channels: channels of signal to measure range count percentage
    :param min_value: desired minimum value
    :param max_value: desired maximum value
    :return: dataframe of calculated range count percentage for each signal channel
    '''
    range_count_df = pd.DataFrame()

    for channel in channels:
        signal_x = signal_df[channel]
        current_range_count = tsf.feature_extraction.feature_calculators.range_count(signal_x, min_value, max_value) * 1.0 / len(signal_x)
        range_count_df[channel + '_range_count_per'] = [current_range_count]

    return range_count_df


@jit()
def ratio_beyond_r_sigma(signal_df, channels, r=2):
    '''
    Calculate ratio beyond r signal of sensor signals.

    :param signal_df: dataframe housing desired sensor signals
    :param channels: channels of signal to measure ratio beyond r sigma
    :param r: r sigma
    :return: dataframe of calculated ratio beyond r sigma for each signal channel
    '''
    ratio_beyond_r_sigma_df = pd.DataFrame()

    for channel in channels:
        current_ratio = tsf.feature_extraction.feature_calculators.ratio_beyond_r_sigma(signal_df[channel], r)
        ratio_beyond_r_sigma_df[channel + '_ratio_beyond_r_sigma'] = [current_ratio]

    return ratio_beyond_r_sigma_df


@jit()
def complexity_invariant_distance(signal_df, channels, normalize=True):
    '''
    Calculate complexity invariant distance of sensor signals.

    :param signal_df: dataframe housing desired sensor signals
    :param channels: channels of signal to measure complexity invariant distance
    :param normalize: bool normalize result
    :return: dataframe of calculated complexity invariant distance for each signal channel
    '''
    complexity_distance_df = pd.DataFrame()

    for channel in channels:
        current_distance = tsf.feature_extraction.feature_calculators.cid_ce(signal_df[channel],
                                                                             normalize=normalize)
        complexity_distance_df[channel + '_complexity_invariant_distance'] = [current_distance]

    return complexity_distance_df


@jit()
def autoregressive_coefficients(signal_df, channels, normalize=True):
    '''
    Calculate autoregressive coefficients of sensor signals.

    :param signal_df: dataframe housing desired sensor signals
    :param channels: channels of signal to obtain autoregressive coefficients from
    :param normalize: bool normalize result
    :return: dataframe of autoregressive coefficients for each signal channel
    '''
    ar_coefficient_df = pd.DataFrame()

    for channel in channels:
        current_coeffs = tsf.feature_extraction.feature_calculators.ar_coefficient(signal_df[channel],
                                                                                    param=[{'coeff':1 , 'k': 3},
                                                                                           {'coeff':2 , 'k': 3},
                                                                                           {'coeff':3 , 'k': 3}])

        ar_coefficient_df[channel + '_autoregressive_coeff_1'] = [current_coeffs[0][1]]
        ar_coefficient_df[channel + '_autoregressive_coeff_2'] = [current_coeffs[1][1]]
        ar_coefficient_df[channel + '_autoregressive_coeff_3'] = [current_coeffs[2][1]]

    return ar_coefficient_df


@jit()
def iqr_of_autocovariance(signal_df, channels):
    #TODO: docs
    '''


    :param signal_df: dataframe housing sensor signals
    :param channels: channels of signal to obtain IQR of autocovariance
    :return: dataframe of calculated IQR of autocovariance for each signal channel
    '''
    autocov_range_df = pd.DataFrame()

    n_samples = signal_df.shape[0]
    for channel in channels:
        current_autocov_iqr = stats.iqr(acf(signal_df[channel], unbiased=True, nlags=n_samples/2))
        autocov_range_df[channel + '_iqr_of_autocovariance'] = [current_autocov_iqr]

    return autocov_range_df

def tsfresh_features(signal_df, channels):
    '''
    Calculate features of sensor signal using TSFresh package.

    :param signal_df: dataframe housing sensor signals
    :param channels: channels of sensor signal to calculate TSFresh features
    :return: dataframe of calculated features for each sensor channel
    '''

    signal_df = signal_df[channels]
    signal_df.loc[:, 'id'] = 1
    tsfresh_df = tsf.extract_features(signal_df, column_id='id', disable_progressbar=True)

    return tsfresh_df.reset_index(drop=True)

def jerk_metric(signal_df, sampling_rate, channels):
    '''
    Calculate jerk of sensor signals.

    :param signal_df: dataframe housing desired sensor signals
    :param sampling_rate: sampling rate of sensor signals
    :param channels: channels of sensor signal to compute jerk
    :return: dataframe of calculated jerk for each sensor channel
    '''
    jerk_ratio_df = pd.DataFrame()

    dt = 1. / sampling_rate
    duration = len(signal_df) * dt

    for channel in channels:
        amplitude = max(abs(signal_df[channel]))

        jerk = signal_df[channel].diff(1) / dt
        jerk_squared = jerk ** 2
        jerk_squared_sum = jerk_squared.sum(axis=0)
        scale = 360 * amplitude ** 2 / duration

        mean_squared_jerk = jerk_squared_sum * dt / (duration * 2)

        jerk_ratio_df[channel + '_jerk_ratio'] = [mean_squared_jerk / scale]

    return jerk_ratio_df


def dimensionless_jerk(signal, sampling_rate, signal_type='velocity'):
    """
    Calculates the smoothness metric for the given movement data using the
    dimensionless jerk metric. The input movement data can be 'speed',
    'accleration' or 'jerk'.
    Parameters
    ----------
    signal : np.array
               The array containing the movement speed profile.
    sampling_rate       : float
               The sampling frequency of the data.
    signal_type: string
               The type of movement data provided. This will determine the
               scaling factor to be used. There are only three possibiliies,
               {'speed', 'accl', 'jerk'}
    Returns
    -------
    dl       : float
               The dimensionless jerk estimate of the given movement's
               smoothness.
    Notes
    -----
    Examples
    --------
    # >>> t = np.arange(-1, 1, 0.01)
    # >>> move = np.exp(-5*pow(t, 2))
    # >>> dl = dimensionless_jerk(move, fs=100.)
    # >>> '%.5f' % dl
    '-335.74684'
    """
    # first ensure the movement type is valid.
    if signal_type in ('velocity', 'acceleration', 'jerk'):
        # first enforce data into an numpy array.
        signal = np.array(signal)

        # calculate the scale factor and jerk.
        movement_peak = max(abs(signal))
        dt = 1. / sampling_rate
        movement_dur = len(signal) * dt
        # get scaling factor:
        _p = {'velocity': 3,
              'acceleration': 1,
              'jerk': -1}
        p = _p[signal_type]
        scale = pow(movement_dur, p) / pow(movement_peak, 2)

        # estimate jerk
        if signal_type == 'velocity':
            jerk = np.diff(signal, 2) / pow(dt, 2)
        elif signal_type == 'acceleration':
            jerk = np.diff(signal, 1) / pow(dt, 1)
        else:
            jerk = signal

        # estimate dj
        return - scale * sum(pow(jerk, 2)) * dt
    else:
        raise ValueError('\n'.join(("The argument data_type must be either",
                                    "'velocity', 'acceleration' or 'jerk'.")))


def log_dimensionless_jerk(signal_df, sampling_rate, channels, signal_type='speed'):
    """
    Calculates the smoothness metric for the given movement data using the
    log dimensionless jerk metric. The input movement data can be 'speed',
    'accleration' or 'jerk'.
    Parameters
    ----------
    signal_df : pandas data frame
                The data frame containing the movement data signals.
    sampling_rate : float
                    The sampling frequency of the data.
    signal_type : string
                  The type of movement data provided. This will determine the
                  scaling factor to be used. There are only three possibiliies,
                  {'speed', 'accl', 'jerk'}
    Returns
    -------
    log_dim_less_jerk_df  : pandas data frame
                            The log dimensionless jerk estimate of the given movement's
                            smoothness for each movement signal.
    Notes
    -----
    Examples
    --------
    # >>> t = np.arange(-1, 1, 0.01)
    # >>> move = np.exp(-5*pow(t, 2))
    # >>> ldl = log_dimensionless_jerk(move, fs=100.)
    # >>> '%.5f' % ldl
    '-5.81636'
    """
    log_dim_less_jerk_df = pd.DataFrame()

    for channel in channels:
        log_dim_less_jerk_df[channel + '_ldl_jerk'] = [-np.log(abs(dimensionless_jerk(signal_df[channel], sampling_rate, signal_type)))]

    return log_dim_less_jerk_df


def sparc(signal, sampling_rate, padlevel=4, fc=10.0, amplitude_threshold=0.05):
    """
    Calcualtes the smoothness of the given speed profile using the modified
    spectral arc length metric.
    Parameters
    ----------
    signal : np.array
               The array containing the movement speed profile.
    sampling_rate  : float
                     The sampling frequency of the data.
    padlevel : integer, optional
               Indicates the amount of zero padding to be done to the movement
               data for estimating the spectral arc length. [default = 4]
    fc       : float, optional
               The max. cut off frequency for calculating the spectral arc
               length metric. [default = 10.]
    amplitude_threshold   : float, optional
                            The amplitude threshold to used for determing the cut off
                            frequency upto which the spectral arc length is to be estimated.
                            [default = 0.05]
    Returns
    -------
    sal      : float
               The spectral arc length estimate of the given movement's
               smoothness.
    (f, Mf)  : tuple of two np.arrays
               This is the frequency(f) and the magntiude spectrum(Mf) of the
               given movement data. This spectral is from 0. to fs/2.
    (f_sel, Mf_sel) : tuple of two np.arrays
                      This is the portion of the spectrum that is selected for
                      calculating the spectral arc length.
    Notes
    -----
    This is the modfieid spectral arc length metric, which has been tested only
    for discrete movements.
    Examples
    --------
    # >>> t = np.arange(-1, 1, 0.01)
    # >>> move = np.exp(-5*pow(t, 2))
    # >>> sal, _, _ = sparc(move, fs=100.)
    # >>> '%.5f' % sal
    '-1.41403'
    """
    # Number of zeros to be padded.
    nfft = int(pow(2, np.ceil(np.log2(len(signal))) + padlevel))

    # Frequency
    f = np.arange(0, sampling_rate, sampling_rate / nfft)

    # Normalized magnitude spectrum
    Mf = abs(np.fft.fft(signal, nfft))
    Mf = Mf / max(Mf)

    # Indices to choose only the spectrum within the given cut off frequency
    # Fc.
    # NOTE: This is a low pass filtering operation to get rid of high frequency
    # noise from affecting the next step (amplitude threshold based cut off for
    # arc length calculation).
    fc_inx = ((f <= fc) * 1).nonzero()
    f_sel = f[fc_inx]
    Mf_sel = Mf[fc_inx]

    # Choose the amplitude threshold based cut off frequency.
    # Index of the last point on the magnitude spectrum that is greater than
    # or equal to the amplitude threshold.
    inx = ((Mf_sel >= amplitude_threshold) * 1).nonzero()[0]
    fc_inx = range(inx[0], inx[-1] + 1)
    f_sel = f_sel[fc_inx]
    Mf_sel = Mf_sel[fc_inx]

    # Calculate arc length
    new_sal = -sum(np.sqrt(pow(np.diff(f_sel) / (f_sel[-1] - f_sel[0]), 2) +
                           pow(np.diff(Mf_sel), 2)))

    return new_sal


def get_sparc_measure(signal_df, sampling_rate, channels, padlevel=4, fc=10.0, amplitude_threshold=0.05):
    '''
    Function to calculate sparc measure for movement signals
    :param signal_df: Data frame with movement signals as columns
    :param sampling_rate: Sampling rate of the movement signals
    :param channels: Channels for which sparc measure will be 
    :param padlevel: Indicates the amount of zero padding to be done to the movement
                     data for estimating the spectral arc length. [default = 4]
    :param fc: The max. cut off frequency for calculating the spectral arc
               length metric. [default = 10.]
    :param amplitude_threshold: The amplitude threshold to used for determing the cut off
                                frequency upto which the spectral arc length is to be estimated.
                                [default = 0.05]
    :return: sparc_df: Data frame containing sparc measure for each channel
    '''
    sparc_df = pd.DataFrame()

    for channel in channels:
        sparc_df[channel + '_sparc'] = [sparc(signal_df[channel],
                                             sampling_rate,
                                             padlevel=padlevel,
                                             fc=fc,
                                             amplitude_threshold=amplitude_threshold)]

    return sparc_df


def get_recurrence_network_measures(signal_df, channels):
    '''
    Calculate recurrence network measures of sensor signals.

    :param signal_df: dataframe housing desired sensor signals
    :param channels: channels of sensor signal to compute recurrence network measures
    :return: dataframe of calculated recurrence network measures for each signal channel
    '''
    recurrence_df = pd.DataFrame()

    for channel in channels:
        time_series = signal_df[channel].values

        # Standardize the time series so it has zero mean and unit variance
        time_series = (time_series - np.mean(time_series)) / np.std(time_series)

        rn = RecurrenceNetwork(time_series, threshold=0.5, silence_level=10)

        # Build the dataframe with features derived from the recurrence network
        recurrence_df[channel + '_average_diagonal_line_length'] = [rn.average_diaglength()]
        recurrence_df[channel + '_average_vertical_line_length'] = [rn.average_vertlength()]
        recurrence_df[channel + '_average_white_vertical_line_length'] = [rn.average_white_vertlength()]
        recurrence_df[channel + '_determinism'] = [rn.determinism()]
        recurrence_df[channel + '_diagonal_line_entropy'] = [rn.diag_entropy()]
        recurrence_df[channel + '_laminarity'] = [rn.laminarity()]
        recurrence_df[channel + '_max_diagonal_line_length'] = [rn.max_diaglength()]
        recurrence_df[channel + '_max_vertical_line_length'] = [rn.max_vertlength()]
        recurrence_df[channel + '_max_white_vertical_line_length'] = [rn.max_white_vertlength()]
        recurrence_df[channel + '_recurrence_probability'] = [rn.recurrence_probability()]
        recurrence_df[channel + '_vertical_line_entropy'] = [rn.vert_entropy()]
        recurrence_df[channel + '_white_vertical_line_entropy'] = [rn.white_vert_entropy()]

        recurrence_df[channel + '_average_path_length'] = [rn.average_path_length()]
        recurrence_df[channel + '_assortativity'] = [rn.assortativity()]
        recurrence_df[channel + '_transitivity'] = [rn.transitivity()]

    return recurrence_df


def signal_envelope_slope(signal_df, sampling_rate, channels):
    '''
    Calculate signal envelope slope of sensor signals.

    :param signal_df: dataframe housing desired sensor signals
    :param sampling_rate: sampling rate of sensor signal
    :param channels: channels of sensor signal to compute signal envelope slope
    :return: dataframe of calculated envelope slop for each signal channel
    '''
    envelope_slope_df = pd.DataFrame()

    num_samples = signal_df.shape[0]
    t = np.linspace(0, num_samples*1.0/sampling_rate, num_samples)

    for channel in channels:
        # rms_df[channel + '_rms'] = [np.std(signal_df[channel] - signal_df[channel].mean())]
        envelope_slope_df[channel + '_envelope_slope'] = get_slope(t, signal_df[channel], method='robust')

    return envelope_slope_df


def get_slope(x, y, method='linear'):
    '''
    Calculate slope from fitted linear regression model of sensor signals.

    :param x: X values used as input to fit linear regression model
    :param y: Y values used as input to fit linear regression model
    :param method: desired method of linear regression model fitting ('linear' or 'robust')
    :return: computed slope of fitted linear regression model
    '''
    if method is 'linear':
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    elif method is 'robust':
        model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
        model_ransac.fit(x.reshape(-1, 1), y)
        slope = model_ransac.estimator_.coef_
        intercept = model_ransac.estimator_.intercept_
    else:
        return

    return slope

def get_frequency_spectrum(signal_x, sampling_rate, cutoff=12):
    '''
    Get the frequency spectrum of a signal
    :param signal_x: 1D signal (numpy array)
    :param sampling_rate: Sampling rate (Hz)
    :param cutoff: Cutoff frequency (Hz)
    :return: Tuple of frequency values and magnitude values corresponding to those frequencies
    '''
    padfactor = 1
    dim = signal_x.shape
    nfft = 2 ** ((dim[0] * padfactor).bit_length())

    freq_hat = np.fft.fftfreq(nfft) * sampling_rate
    freq = freq_hat[0:nfft / 2]

    idx1 = freq <= cutoff
    idx_cutoff = np.argwhere(idx1)
    freq = freq[idx_cutoff]
    freq = freq.flatten()

    sp_hat = np.fft.fft(signal_x, nfft)
    sp = sp_hat[0:nfft / 2] * np.conjugate(sp_hat[0:nfft / 2])
    sp = sp[idx_cutoff]
    sp_norm = sp / sum(sp)
    sp_norm = abs(sp_norm.flatten())

    return (freq, sp_norm)

def extract_frequency_spectrum_features(sp_norm, freq):
    freq_offset = 0.25

    sp_norm = sp_norm / sp_norm.sum()

    # Dominant frequency and its magnitude in the 0.25 - 3 Hz band (LF)
    sp_norm_lf = sp_norm[freq < 3]
    freq_lf = freq[freq < 3]

    lf_dom_freq_magnitude = sp_norm_lf.max()
    lf_dom_freq = freq_lf.item(sp_norm_lf.argmax())

    lf_dom_freq_energy = sp_norm[np.argwhere((freq <= lf_dom_freq + freq_offset) & (freq >= lf_dom_freq - freq_offset))].flatten().sum()
    lf_dom_freq_ratio = lf_dom_freq_energy / sp_norm_lf.sum()

    # Dominant frequency and its magnitude in the 3 - 8 Hz band (HF)
    sp_norm_hf = abs(sp_norm[np.argwhere((freq <= 8) & (freq >= 3))].flatten())
    freq_hf = freq[np.argwhere((freq <= 8) & (freq >= 3))].flatten()

    hf_dom_freq_magnitude = sp_norm_hf.max()
    hf_dom_freq = freq_hf.item(sp_norm_hf.argmax())

    hf_dom_freq_energy = sp_norm[np.argwhere((freq <= hf_dom_freq + freq_offset) & (freq >= hf_dom_freq - freq_offset))].flatten().sum()
    hf_dom_freq_ratio = hf_dom_freq_energy / sp_norm_hf.sum()

    # Ratio of energy in the HF to LF band
    hf_lf_ratio = sp_norm_hf.sum() / sp_norm_lf.sum()

    feature_dict = {'lf_dominant_frequency_value': lf_dom_freq,
                    'lf_dominant_frequency_magnitude': lf_dom_freq_magnitude,
                    'lf_dominant_frequency_energy': lf_dom_freq_energy,
                    'lf_dominant_frequency_ratio': lf_dom_freq_ratio,
                    'hf_dominant_frequency_value': hf_dom_freq,
                    'hf_dominant_frequency_magnitude': hf_dom_freq_magnitude,
                    'hf_dominant_frequency_energy': hf_dom_freq_energy,
                    'hf_dominant_frequency_ratio': hf_dom_freq_ratio,
                    'hf_lf_energy_ratio': hf_lf_ratio
                    }

    feature_df = pd.DataFrame(feature_dict, index=[0])

    return feature_df

@jit()
def activity_index(signal_df, channels=['X', 'Y', 'Z']):
    """
    Compute activity index of sensor signals.

    :param signal_df: dataframe housing desired sensor signals
    :param channels: channels of signal to compute activity index
    :return: dataframe housing calculated activity index
    """
    ai_df = pd.DataFrame()
    ai_df['activity_index'] = [np.var(signal_df[channels], axis=0).mean() ** 0.5]
    return ai_df

@jit()
def signal_std(signal_df, channels):
    """
    Compute std of sensor signals.

    :param signal_df: dataframe housing desired sensor signals
    :param channels: channels of signal to measure std
    :return: dataframe housing calculated std for each signal channel
    """
    std_df = pd.DataFrame()

    for channel in channels:
        std_df[channel + '_std'] = [signal_df[channel].std()]

    return std_df

@jit()
def max_3_axis_std(signal_df, channels=['X', 'Y', 'Z']):
    """
    Compute max std of sensor signals.

    :param signal_df: dataframe housing raw sensor signals
    :param channels: channels of signal to measure std
    :return: dataframe housing max std across all channels
    """
    max_3_axis_std_df = pd.DataFrame()

    max_3_axis_std_df['max_3_axis_std'] = [signal_df[channels].std().max(skipna=True)]

    return max_3_axis_std_df

@jit()
def max_3_axis_range(signal_df, channels=['X', 'Y', 'Z']):
    """
    Compute max range of sensor signals.

    :param signal_df: dataframe housing raw sensor signals
    :param channels: channels of signal to measure range
    :return: dataframe housing max value range across all channels
    """
    max_3_axis_range_df = pd.DataFrame()

    max_3_axis_range_df['max_3_axis_range'] = [np.max((signal_df[channels].max(skipna=True) -
                                                       signal_df[channels].min(skipna=True)))]

    return max_3_axis_range_df
