import numpy as np
from math import floor, ceil


def amplify(signal, dB):
    """
    Amplify or attenuate signal in terms of power by specified amount in decibels

    Arguments:
        signal (ndarray): input sound pressure signal
        dB (float): dB amount to amplify by
    Returns:
        output (array): output sound pressure signal
    """
    # Determine scale factor and then apply to signal
    scale_factor = 10**(dB/20)
    return signal * scale_factor


def complex_tone(freqs, levels, phases, dur, fs):
    """
    Synthesize complex tone as series of pure tones

    Arguments:
        freqs (ndarray): harmonic frequencies in Hz
        levels (ndarray): levels of each harmonic in dB SPL
        phases (ndarray): phase offsets of each harmonic in degrees
        dur (float): duration in seconds
        fs (int): sampling rate in Hz

    Returns:
        output (array): complex tone
    """
    # Check to make sure argument lengths match up
    if not (len(freqs) == len(levels) and len(freqs) == len(phases)):
        raise Exception("Length of freqs, amps, and phases should match")
    # Create empty output vector
    output = np.zeros((floor(dur*fs)))
    # Synthesize each component pure tone and add to output vector
    for (freq, amp, phase) in zip(freqs, levels, phases):
        output = output + scale_dbspl(pure_tone(freq, phase, dur, fs), amp)
    return output


def cosine_ramp(signal, dur_ramp, fs):
    """
    Applies a raised-cosine ramp to a time-domain signal.

    Arguments:
        signal (np.ndarray): time-domain signal to be ramped
        dur_ramp (float): duration of the ramp in seconds
        fs (int): sampling rate in Hz

    Returns:
        output (array): time-domain signal with ramp
    """
    # Determine length of the ramp
    n_ramp = floor(fs*dur_ramp)
    # Calculate shape of ramp
    ramp = np.sin(np.linspace(0, np.pi/2, n_ramp))**2
    # Combine ramp with flat middle and return
    ramp = np.concatenate((ramp, np.ones((len(signal)-n_ramp*2)),
                           np.flip(ramp)))
    return signal*ramp


def dbspl_pascal(signal):
    """
    Returns the expected dB SPL level of a time-domain signal

    Arguments:
        signal (ndarray): input sound pressure signal

    Returns:
        output (float): dB SPL of signal
    """
    # Check to make sure that our signal is not all zeros
    if rms(signal) == 0:
        raise ValueError('RMS of signal iz zero and has no dB value.')
    # Calibrate reference at 20 micropascals
    ref = 20e-6
    # Calculate and return dB SPL value
    return 20*np.log10(rms(signal)/ref)


def pure_tone(freq, phase, dur, fs):
    """
    Synthesize single pure tone

    Arguments:
        freq (float): frequency of pure tone in Hz
        phase (float): phase offset in degrees, must be between 0 and 360
        dur (float): duration in seconds
        fs (int): sampling rate in Hz

    Returns:
        output (array): pure tone
    """
    # Create empty array of time samples
    t = np.linspace(0, dur, floor(dur*fs))
    # Calculate and return pure tone
    return np.sin(2*np.pi*freq*t+(2*np.pi/360*phase))


def rms(signal):
    """
    Computes root-mean-square (RMS) of signal

    Arguments:
        signal (ndarray): time-domain signal

    Returns:
        output (float): RMS value of signal
    """
    return np.sqrt(np.mean(signal**2))


def scale_dbspl(signal, dB):
    """
    Scale time domain signal to have certain level in dB SPL

    Arguments:
        signal (ndarray): input sound pressure signal
        dB (float): desired level in dB SPL

    Returns:
        val (array): output sound pressure signal
    """
    # Calculate current dB SPL value and then calculate differential and apply to signal
    curr_dB = dbspl_pascal(signal)
    delta_dB = dB - curr_dB
    return amplify(signal, delta_dB)