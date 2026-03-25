# Function to compute the Continuous Wavelet Transform (CWT)
# Developed for the manuscript:
# 'Enhanced Time-Frequency Analysis of Surface Electromyography using Superlet Transform'
# by Algaba-Vidoy et al., 2026 (under revision)
#
# In this implementation, the CWT is computed in the time domain:
# scales are directly derived from the target frequency range, and
# the correlation of each wavelet with the signal is evaluated.
#
# Implementation by Marina Algaba Vidoy


import pywt


def wavelet_transform(signal, wavelet, freqs, sampling_frequency):
    """
    Computes the Continuous Wavelet Transform (CWT) of a signal using specified wavelet and frequencies.

    Parameters
        signal : array, time-domain signal.
        wavelet : str, wavelet function name or object to use for the transform (e.g., 'morl').
        freqs : array, frequencies of interest (in Hz) for which to compute the CWT.
        sampling_frequency : float, sampling frequency of the input signal (in Hz).

    Returns
        cwtmatr : 2D array, CWT coefficients matrix (scales x time).
        f : array, corresponding scales used in the transform.
        physical_freqs : array, frequencies corresponding to the scales (in Hz).
    """
    sampling_period = 1 / sampling_frequency
    # Compute the scales from the frequencies of interest
    scales = pywt.frequency2scale('morl', freqs, precision=10) / sampling_period
    # Represented frequency with this scales
    physical_freqs = pywt.scale2frequency('morl', scale=scales, precision=10) / sampling_period
    # Compute the CWT
    cwtmatr, f = pywt.cwt(signal, scales, wavelet, sampling_period)

    return cwtmatr, f, physical_freqs