# Auxiliary functions for the assessment of superlets in sEMG analysis.
# Developed for the manuscript:
# 'Enhanced Time-Frequency Analysis of Surface Electromyography using Superlet Transform'
# by Algaba-Vidoy et al., 2026 (under revision)
#
# These functions are intended to support the evaluation and validation
# of superlet-based time-frequency representations applied to sEMG signals,
# including performance metrics, spectral feature extraction, signal quality measures,
# and the generation of the simulated signals.
#
# Implementation by Marina Algaba Vidoy
#
# Note: These utilities are designed to complement the main analysis pipeline.


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.fft import ifft
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d


def periodogram_max_res(data, fs, plot=False):
    """
    Computes the maximum-resolution periodogram for short signals.

    Parameters
        data: array, time-domain signal.
        fs: float, sampling frequency in Hz.
        plot: bool (optional), if True, plots the power spectral density (PSD) and frequency markers (default is False).

    Returns
        pxx: array, Power Spectral Density (PSD).
        f: array, frequencies in Hz.
        mnf: float, Mean Frequency.
    """
    l = len(data)
    nfft = l
    noverlap = 0
    window = np.hanning(nfft)

    pxx, f = plt.psd(data, NFFT=nfft, Fs=fs, window=window, noverlap=noverlap)
    plt.close()

    delta_f = fs / nfft
    print(f"Achieved frequency resolution (Δf): {delta_f:.2f} Hz")

    mnf = meanfreq(pxx, f)
    mdf = medfreq(pxx, f)

    if plot:
        fig, ax = plt.subplots()

        ax.plot(f, pxx)
        ax.set(title=' PSD', xlabel='Frequency (Hz)', ylabel='Power (dB)')

        ax.axvline(x=mdf, color='orange', linestyle='--',
                    label=f'Median Frequency (MDF) = {mdf:.2f}')

        ax.axvline(x=mnf, color='green', linestyle='--',
                    label=f'Mean Frequency (MNF) = {mnf:.2f}')

        ax.legend()

    return pxx, f, mnf


def meanfreq(pxx, f):
    """
    Computes the Mean Frequency (mnf) of a power spectral density.

    Parameters
        pxx: array, Power Spectral Density (PSD).
        f: array, frequency vector corresponding to the pxx.

    Returns
        mnf: float, mean frequency.
    """
    # Compute the width of each frequency bin
    width = np.diff(f, prepend=f[0])

    # Power contribution of each frequency bin
    p = width * pxx

    freq_range = [f[0], f[-1]]
    idx = (freq_range[0] <= f) & (f <= freq_range[1])

    # Total power within the range
    pidx = p[idx]
    pwr = np.sum(pidx)

    mnf = np.sum(pidx * f[idx]) / pwr

    return mnf


def medfreq(pxx, f):
    """
    Computes the median frequency (mdf) of a power spectral density.

    Parameters
        pxx: array, Power Spectral Density (PSD).
        f: array, frequency vector corresponding to the pxx.

    Returns
        mdf: float, median frequency.
    """
    # Compute the width of each frequency bin
    width = np.diff(f, prepend=f[0])

    # Power contribution of each frequency bin
    p = width * pxx

    freq_range = [f[0], f[-1]]
    idx = (freq_range[0] <= f) & (f <= freq_range[1])

    # Total power within the range
    pidx = p[idx]
    pwr = np.sum(pidx)

    # Compute the mdf (half of the total power)
    cumulative_power = np.cumsum(pidx)
    half_power = pwr / 2
    median_idx = np.where(cumulative_power >= half_power)[0][0]

    mdf = f[idx][median_idx]

    return mdf


def fdeluca(fh, fl, fc, t, plot=False):
    """
    Generates a De Luca filter using the Stulen & De Luca 1980 formula and simulates muscle contraction
    by filtering a series of white Gaussian noise using this filter.

    Parameters
        fh: float, upper cutoff frequency of the filter (Hz).
        fl: float, lower cutoff frequency of the filter (Hz).
        fc: float, sampling frequency (Hz).
        t: float, duration of the simulated signal (s).
        plot: bool (optional), if True, intermediate spectra and filter responses are plotted (default is False).

    Returns
        filter: array, time-domain response of the De Luca filter (samples of the filter applied to white noise).
        pdeluca: array, the simulated power spectral density (PSD).
        emg_sim: array, the simulated EMG signal.
        mnf_ideal: float, mean frequency (mnf) of the ideal De Luca spectrum.
        mnf_analytic: float, analytically derived mean frequency.
    """

    # Frequency range for the filter design
    f = np.arange(1, fc / 2 + 1)

    # Power spectral density (PSD) based on De Luca's formula (power is normalized)
    pdeluca = (fh ** 4 * f ** 2) / ((f ** 2 + fl ** 2) * ((f ** 2 + fh ** 2) ** 2))

    # The PSD array for both positive and negative frequencies
    m = fc // 2  # Central sample
    px = np.concatenate([pdeluca, pdeluca[m - 1:m], np.flipud(pdeluca)])

    # Convert the PSD into a time-domain filter using the Inverse FFT
    px = np.sqrt(px)

    # Apply the Hilbert transform to get phase information
    px1 = px * np.exp(-1j * np.imag(hilbert(np.log(px))))

    mnf_ideal = meanfreq(pdeluca, f)
    mdf_ideal = medfreq(pdeluca, f)

    plt.figure()
    plt.plot((pdeluca / (max(pdeluca))), label='PSD')
    plt.axvline(x=mdf_ideal, color='orange', linestyle='--',
                label=f'Median Frequency (MDF) = {mdf_ideal:.2f} Hz')

    # Línea vertical en la frecuencia media (MNF)
    plt.axvline(x=mnf_ideal, color='green', linestyle='--',
                label=f'Mean Frequency (MNF) = {mnf_ideal:.2f} Hz')

    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized power')
    plt.title('Power Spectrum')

    envelope = np.abs(px1)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(px, label='Original Signal (Px)')
        plt.plot(envelope, label='Envelope (Hilbert)', linestyle='--')
        plt.title(f'Envelope and Signal')
        plt.xlabel('Frequency Bin')
        plt.ylabel('Amplitude')
        plt.legend(framealpha=1, shadow=True)
        plt.grid()
        plt.show()

    # The filter response in the time domain (Inverse FFT of Px1)
    filter = np.real(ifft(px1, fc))

    if plot:
        plt.figure()
        plt.plot(filter, label=f'Raw filter')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.title('Original Filter')
        plt.grid()
        plt.legend()
        plt.show()

    # Normalize the filter response
    pdeluca = pdeluca / len(pdeluca)

    # Generate white Gaussian noise
    n = np.random.randn(int(t * fc) + 128)

    # Simulate the EMG signal by filtering the white noise with the De Luca filter
    emg_sim = lfilter(filter, 1, n)
    # Discard the first 129 samples to avoid filter edge effects
    emg_sim = emg_sim[129:]

    # Calculate the Alpha factor
    alpha = fl / fh

    # Uncomment and implement MNF_analytic if needed for additional analysis
    mnf_analytic = (2 * fh / np.pi) * ((1 + alpha) / (1 - alpha)) * (1 - (2 * alpha**2) / (alpha**2 - 1) * np.log(alpha))

    return filter, pdeluca, emg_sim, mnf_ideal, mnf_analytic


def compute_snr(signal, noise):
    """
    Computes the Signal-to-Noise Ratio (SNR) in decibels (dB).

    Parameters
        signal : array-like, the original signal.
        noise : array-like, the noise array.

    Returns
        snr_db : float, signal-to-noise ratio expressed in decibels (dB).
    """
    # Power of the signal
    signal_power = np.mean(signal ** 2)
    # Power of the noise
    noise_power = np.mean(noise ** 2)

    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


def add_noise(x, snr, fs, plot=False, seed=10):
    """
    Add Gaussian random noise to a signal with a specified SNR.

    Parameters
        x: numpy array, original signal.
        snr: float, desired Signal-to-Noise Ratio in dB.
        fs: float, sampling frequency of the signal in Hz.
        plot: bool (optional), if True plots the original and noisy signals (default is False).
        seed: int (optional), random seed for noise generation to ensure reproducibility (default is 10).

    Returns
        xn: array, noisy signal with the specified SNR.
        noise: array, generated Gaussian noise added to the signal.
    """

    n = len(x)  # Length of the signal
    Px = np.sum(np.abs(x)**2) / n  # Power of the original signal
    Pn = Px / (10**(snr / 10))  # Noise power for the given SNR

    np.random.seed(seed)

    noise = np.sqrt(Pn) * np.random.randn(*x.shape)  # Generate Gaussian noise
    xn = x + noise  # Add noise to the original signal

    t = np.linspace(0, len(noise)/fs, len(noise))

    # Plot the original and noisy signals
    if plot:
        plt.figure()
        plt.plot(t, xn, 'r', label='Noisy Signal')
        plt.plot(t, x, 'b', label='Original Signal')
        plt.legend()
        plt.title(f'Signal with known burst (SNR = {snr} dB)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (mV)')
        plt.show()

    return xn, noise


def find_plateau_region(signal, burst_start, burst_duration, t, fs, smooth_sigma=20, std_threshold=None, plot_results=False):
    """
    Finds the start and end indices of a plateau region in a signal.

    Parameters
        signal: array, input signal.
        burst_start: float, start time of the burst in seconds.
        burst_duration: float, duration of the burst in seconds.
        t: array, time vector corresponding to the signal.
        fs: int, sampling frequency in Hz.
        smooth_signal: float (optional), sigma of the Gaussian filter used to smooth the signal (default is 20).
        std_threshold: float (optional), threshold for plateau detection, defined as a fraction of max signal amplitude (default is 20% of max).
        plot_results: bool (optional), if True, plots the signal, smoothed signal, threshold, and detected plateau region (default is False).

    Returns
        plateau_start_index : int, index where the plateau starts.
        plateau_end_index : int, index where the plateau ends.
    """

    # Smooth input signal
    smooth_signal = gaussian_filter1d(signal, sigma=smooth_sigma)

    # Determine threshold automatically if not provided
    signal_max = np.max(smooth_signal)
    if not std_threshold:
        std_threshold = 0.2 * signal_max

    # Calculate center index and time window around the burst
    center = round(((burst_duration / 2) + burst_start) * fs)
    time_shift = round(burst_duration * 2 * fs)
    start_idx = max(0, center - time_shift)
    end_idx = min(len(t), center + time_shift)

    # Identify candidate plateau indices where smoothed signal exceeds threshold
    plateau_candidates = np.where(smooth_signal[start_idx:end_idx] > std_threshold)[0] + start_idx

    if len(plateau_candidates) > 0:
        plateau_start_index = plateau_candidates[0]
        plateau_end_index = plateau_candidates[-1]
    else:
        plateau_start_index = start_idx
        plateau_end_index = end_idx

    if plot_results:
        plt.figure(figsize=(10, 4))
        plt.plot(signal)
        plt.plot(smooth_signal, label="Smoothed instantaneous frequency", linewidth=2)
        plt.axhline(std_threshold, color='gray', label='Threshold')

        plt.title("Energy over time")

        plt.axhline(signal_max, color='purple', linestyle='--', label="Max")

        plt.legend()
        plt.axvline(plateau_start_index, color='green', linestyle='--', label="Centro")
        plt.axvline(plateau_end_index, color='red', linestyle='--', label="Centro")

    return plateau_start_index, plateau_end_index


def calculate_mae(estimates, true_value, plot=False):
    """
    Calculates the Mean Absolute Error (MAE) between estimates and true values.

    Parameters
        estimates: array-like or tuple of arrays, estimated values. It can be a single array or a tuple of two arrays (e.g., start and end estimates).
        true_value: scalar or tuple of scalars, true value(s). Can be a single scalar or a tuple of two scalars (e.g., true start and end).
        plot: bool (optional), if True, plots the absolute errors (default is False).

    Returns
        mae : float, Mean Absolute Error (mae).
        std_error : float, standard deviation of the errors.
    """
    estimates = np.array(estimates)

    # Case: single true value or one-element array
    if np.isscalar(true_value) or len(true_value) == 1:
        errors = np.abs(estimates - true_value)
        if plot:
            plt.figure()
            plt.plot(errors)
            plt.axhline(np.mean(errors))

    # Case: tuple/list with start and end values
    else:
        estimated_starts, estimated_ends = estimates
        true_start, true_end = true_value
        errors_start = np.abs(estimated_starts - true_start)
        errors_end = np.abs(estimated_ends - true_end)
        errors = [errors_start, errors_end]

    mae = np.mean(errors)
    std_error = np.std(errors)
    return mae, std_error

def calculate_fwhm(response):
    """
    Calculates the Full Width at Half Maximum (FWHM) of a 1D response array.

    Parameters
        response: array, input signal.

    Returns
        fwhm: float, FWHM
        x1: int, index of the first crossing point above half maximum.
        x2: int, index of the last crossing point above half maximum.
        half_max: float, value of half the maximum  of the response.
    """
    half_max = np.max(response) / 2

    above_half_max = np.where(response > half_max)[0]

    # If there are multiple points, FWHM is distance between first and last crossing
    x1, x2 = above_half_max[0], above_half_max[-1]
    fwhm = x2 - x1
    return fwhm, x1, x2, half_max