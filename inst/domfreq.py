import numpy as np
import pandas as pd
from scipy import stats

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
        freq = freq_hat[0:nfft // 2]

        idx1 = freq <= cutoff
        idx_cutoff = np.argwhere(idx1)
        freq = freq[idx_cutoff]

        sp_hat = np.fft.fft(signal_x, nfft)
        sp = sp_hat[0:nfft // 2] * np.conjugate(sp_hat[0:nfft // 2])
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
