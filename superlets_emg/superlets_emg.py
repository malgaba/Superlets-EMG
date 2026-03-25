# Function to compute the Superlet Transform adapted to sEMG analysis
# Developed for the manuscript:
# 'Enhanced Time-Frequency Analysis of Surface Electromyography using Superlet Transform'
# by Algaba-Vidoy et al., 2026 (under revision)
#
# Implementation by Marina Algaba Vidoy based on the original implementations by:
# - Harald Bârzan and Richard Eugen Ardelean
#   (https://github.com/TransylvanianInstituteOfNeuroscience/Superlets.git)
# - Irhum Shafkat
#   (https://github.com/irhum/superlets.git)


import numpy as np
from scipy.signal import fftconvolve

# spread, in units of standard deviation, of the Gaussian window of the Morlet wavelet
MORLET_SD_SPREAD = 6

# the length, in units of standard deviation, of the actual support window of the Morlet
MORLET_SD_FACTOR = 2.5


def get_order(f, f_min: int, f_max: int, o_min: int, o_max: int):
    return o_min + np.round((o_max - o_min) * (f - f_min) / (f_max - f_min))


def morlet(fc, nc, fs):
    """
    Create an analytic Morlet wavelet.

    Parameters
        fc: float, center frequency in Hz.
        nc: int, number of cycles.
        fs: int, sampling frequency in Hz.

    Returns
        wavelet : array, complex analytic Morlet wavelet.
    """
    sd = (nc / 2) * (1 / np.abs(fc)) / MORLET_SD_FACTOR
    size = int(2 * np.floor(np.round(sd * fs * MORLET_SD_SPREAD) / 2) + 1)
    half = int(np.floor(size / 2))
    gauss = gausswin(size, MORLET_SD_SPREAD / 2)
    igsum = 1 / gauss.sum()
    ifs = 1 / fs

    t = (np.arange(size, dtype=np.float64) - half) * ifs
    wavelet = gauss * np.exp(2 * np.pi * fc * t * 1j) * igsum

    return wavelet


def gausswin(size, alpha):
    """
    Creates a Gaussian window.

    Parameters
        size : int, length of the window.
        alpha : float, parameter controlling the width (standard deviation) of the Gaussian.

    Returns
        window : array, Gaussian window of the specified size.
    """
    half_size = int(np.floor(size / 2))
    idiv = alpha / half_size

    t = (np.arange(size, dtype=np.float64) - half_size) * idiv
    window = np.exp(-(t * t) * 0.5)

    return window


def adaptive_superlet_transform(signal, freqs, sampling_freq: int, base_cycle: int, min_order: int, max_order: int, mode="mul"):
    """Computes the adaptive superlet transform of the provided signal.

    Parameters
        signal: 1D array containing the signal data.
        freqs: 1D sorted array containing the frequencies to compute the wavelets at.
        sampling_freq: int, sampling frequency of the signal.

        base_cycle: int, number of cycles corresponding to order=1
        min_order: int, minimum upper limit of orders to be used for a frequency in the adaptive superlet.
        max_order: int, maximum upper limit of orders to be used for a frequency in the adaptive superlet.

        mode (str, optional): "add" or "mul", corresponding to the use of additive or multiplicative adaptive superlets (default is "mul").

    Returns:
        out: 2D array (Frequency x Time) representing the computed scalogram
    """

    orders = get_order(freqs, min(freqs), max(freqs), min_order, max_order)

    out = np.zeros((len(freqs), len(signal)))

    for i, fc in enumerate(freqs):
        nWavelets = int(np.ceil(orders[i]))

        acc = np.zeros(len(signal))

        for iWave in range(nWavelets):
            if mode == "mul":
                wavelet = morlet(fc, (iWave + 1) * base_cycle, sampling_freq)
            elif mode == "add":
                wavelet = morlet(fc, (iWave + 1) + base_cycle, sampling_freq)
            else:
                raise ValueError("mode should be one of \"mul\" or \"add\"")

            conv = fftconvolve(signal, wavelet, mode="same")

            acc += np.log(np.abs(conv) ** 2)

        out[i, :] = np.exp(acc / nWavelets)
    return wavelet, out








