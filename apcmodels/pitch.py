import numpy as np


def compute_running_ACF(rates, time_point, fs=int(100e3), tau=10*1e-3, lag_low=1e-3, lag_high=15e-3, n_lag=250):
    """ Computes the summary autocorrelation function (sACF) from Meddis and O'Mard (1991) at a single time point.

    Args:
        rates (ndarray): an array of instantaneous firing rates from an auditory nerve or other neural model, of shape
            (n_channel, n_sample)
        time_point (float): time point (t_0) at which to compute the autocorrelation
        fs (int): sampling rate, in Hz
        tau (float): time constant, in seconds
        lag_low (float): shortest lag at which to compute autocorrelation, in seconds
        lag_high (float): longest lag at which to compute autocorrelation, in seconds
        n_lag (int): number of lag samples to compute

    Returns:
        h (ndarray): autocorrelation values as a function of channel and lag time, of shape (n_channel, n_lag)
        lag_times (ndarray): lag times in seconds, of shape (n_lag, )
    """
    # Create constants
    dt = 1/fs  # sampling period
    t_axis = np.linspace(0, rates.shape[1]/fs, rates.shape[1])  # time axis in seconds
    t_axis_samples = np.arange(rates.shape[1])  # time axis in samples
    # Create lag axis to loop through
    lag = np.round(np.linspace(lag_low, lag_high, n_lag)*fs).astype('int')  # lag times to calculate in samples
    h = np.zeros((rates.shape[0], len(lag)))  # empty array for output, shape (n_channel, n_lag)
    t_0 = int(time_point*fs)  # time point in samples
    # Loop through lags
    for idx_lag, _lag in enumerate(lag):
        h[:, idx_lag] = 1/tau + np.sum(rates[:, t_0-t_axis_samples[0:-(_lag+1)]] *
                                       rates[:, t_0-t_axis_samples[0:-(_lag+1)]-_lag] *
                                       np.exp(-t_axis[0:-(_lag+1)]/tau) * dt, axis=1)
    return h, lag*dt


def estimate_F0_sACF(rates, time_point, fs=int(100e3), tau=10*1e-3, lag_low=1e-3, lag_high=15e-3, n_lag=250):
    """ Estimates the F0 by simple peak-picking summary autocorrelation function from Meddis and O'Mard (1991).

    Args:
        rates (ndarray): an array of instantaneous firing rates from an auditory nerve or other neural model, of shape
            (n_channel, n_sample). It is assumed that these firing rates are in response to some sort of tonal stimulus.
        time_point (float): time point (t_0) at which to compute the autocorrelation
        fs (int): sampling rate, in Hz
        tau (float): time constant, in seconds
        lag_low (float): lowest lag at which to compute autocorrelation, in seconds
        lag_high (float): highest lag at which to compute autocorrelation, in seconds
        n_lag (int): number of lag samples to compute

    Returns:
        F0 (float): estimated F0, in Hz
    """
    h, lags = compute_running_ACF(rates, time_point, fs, tau, lag_low, lag_high, n_lag)
    idx = np.argmax(np.sum(h, axis=0))  # sum ACF across channels/fibers and pick the peak

    return 1/lags[idx]  # return F0


def compute_autocorrelation(rates, fs):
    """ Computes the autocorrelation function (ACF) of a firing-rate simulation.

    Args:
        rates (ndarray): an array of instantaneous firing rates from an auditory nerve or other neural model, of shape
            (n_channel, n_sample)
        fs (int): sampling rate, in Hz

    Returns:
        h (ndarray): autocorrelation values as a function of channel and lag time, of shape (n_channel, n_lag)
        lag_times (ndarray): lag times in seconds, of shape (n_lag, )
    """
    # Create constants
    h = np.real(np.fft.ifft(np.abs(np.fft.fft(rates, axis=1))**2, axis=1))
    lag = np.linspace(0, h.shape[1]/fs, h.shape[1])
    return h, lag


def estimate_F0_autocorrelation(rates, fs, lag_low=1e-04):
    """ Estimates the F0 by simple peak-picking summary autocorrelation function from Meddis and O'Mard (1991).

    Args:
        rates (ndarray): an array of instantaneous firing rates from an auditory nerve or other neural model, of shape
            (n_channel, n_sample). It is assumed that these firing rates are in response to some sort of tonal stimulus.
        fs (int): sampling rate, in Hz
        lag_low (float): shortest lag at which F0 can be evaluated

    Returns:
        F0 (float): estimated F0, in Hz
    """
    h, lags = compute_autocorrelation(rates, fs)
    idx = np.argmax(np.sum(h[:, lags > lag_low], axis=0))  # sum ACF across channels/fibers and pick the peak
    lags = lags[lags > lag_low]

    return 1/lags[idx]  # return F0
