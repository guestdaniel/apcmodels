import numpy as np
from math import floor
from scipy.interpolate import interp1d
from copy import deepcopy


def amplify(signal, dB):
    """
    Amplify or attenuate signal in terms of power by specified amount in decibels

    Arguments:
        signal (ndarray): input sound pressure signal, of shape (n_sample, ) or (n_sample, n_signal)
        dB (float, ndarray): dB amount to amplify by, of shape (1, ) or (n_signal)
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
    # Check to make sure inputs are of the right shape
    if freqs.shape != levels.shape or freqs.shape != phases.shape:
        raise ValueError('Freqs, levels, and phases must all be the same size')
    # Synthesize components in parallel
    output = scale_dbspl(pure_tone(freqs, phases, dur, fs), levels)
    if output.ndim == 1:
        return output  # if we only have one component, return it alone
    else:
        return np.sum(output, axis=1)  # if we have multiple components, sum across them


def cosine_ramp(signal, dur_ramp, fs):
    """
    Applies a raised-cosine ramp to a time-domain signal.

    Arguments:
        signal (np.ndarray): time-domain signal to be ramped, of shape (n_samples, )
        dur_ramp (float): duration of the ramp in seconds
        fs (int): sampling rate in Hz

    Returns:
        output (array): time-domain signal with ramp
    """
    # Check to make sure that the duration of the ramp is less than 1/2 of the stimulus duration
    if dur_ramp > len(signal)/2:
        raise ValueError('The ramp cannot be longer than the stimulus!')
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
        signal (ndarray): input sound pressure signal, either shape of (n_sample, ) or (n_sample, n_signal). In the
            latter case, each signal is processed separately.

    Returns:
        output (float): dB SPL of signal, of shape (1, ) or (n_signal, )
    """
    # Check to make sure that our signal is not all zeros
    if np.any(rms(signal) == 0):
        raise ValueError('RMS of at least one signal in input is zero and has no dB value.')
    # Calibrate reference at 20 micropascals
    ref = 20e-6
    # Calculate and return dB SPL value
    return 20*np.log10(rms(signal)/ref)


def pure_tone(freq, phase, dur, fs):
    """
    Synthesize single pure tone

    Arguments:
        freq (float, ndarray): frequency of pure tone in Hz
        phase (float, ndarray): phase offset in degrees, must be between 0 and 360
        dur (float, ndarray): duration in seconds
        fs (int): sampling rate in Hz

    Returns:
        output (array): pure tone, of shape (n_sample, ) or (n_sample, n_pure_tone)
    """
    # Create empty array of time samples
    t = np.linspace(0, dur, floor(dur*fs))
    # Calculate pure tone
    return np.squeeze(np.sin(2*np.pi*np.outer(t, freq)+(2*np.pi/360*phase)))


def pure_tone_am(freq, phase, freq_mod, phase_mod, depth_mod, dur, fs):
    """
    Synthesize single pure tone with sinusoidal amplitude modulation

    Arguments:
        freq (float): frequency of pure tone in Hz
        phase (float): phase offset in degrees, must be between 0 and 360
        freq_mod (float): frequency of modulator in Hz
        phase_mod (float): phase offset of modulator in Hz
        depth_mod (float): modulation depth in m, bounded from 0 to 1
        dur (float): duration in seconds
        fs (int): sampling rate in Hz

    Returns:
        output (array): pure tone

    TODO: add tests and vectorize
    """
    # Create empty array of time samples
    t = np.linspace(0, dur, floor(dur*fs))
    # Calculate carrier waveform
    carrier = np.sin(2*np.pi*freq*t+(2*np.pi/360*phase))
    # Calculate modulator waveform
    modulator = depth_mod*np.sin(2*np.pi*freq_mod*t+(2*np.pi/360*phase_mod))
    # Return carrier x modulator
    return (1 + modulator) * carrier


def rms(signal, axis=0):
    """
    Computes root-mean-square (RMS) of signal

    Arguments:
        signal (ndarray): time-domain signal, of shape (n_sample, ) or (n_sample, n_signal)
        axis (int): axis along which to calculate RMS, defaults to 0

    Returns:
        output (float): RMS value of signal
    """
    return np.sqrt(np.mean(signal**2, axis=axis))


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


def te_noise(duration, fs, lco, hco, level):
    """
    Synthesizes a sample of threshold-equalizing noise. Adapted with minor changes from code used internally in the
    Auditory Perception and Cognition Lab at the University of Minnesota.

    Arguments:
        duration (float): duration in seconds
        fs (int): sampling rate in Hz
        lco (float): lower cutoff of the noise in Hz
        hco (float): higher cutoff of the noise in Hz
        level (float): level of the noise in the ERB centered on 1 kHz

    Returns:
        noise (ndarray): array of noise of shape (n_sample, )
    """
    # Raise errors if anything is fishy
    if lco < 0:
        raise ValueError('lco should not be below 0 Hz')
    if hco > fs/2:
        raise ValueError('hco should be below Nyquist frequency')

    # Calculate the number of samples in the stimulus
    dur_smp = floor(duration * fs)

    # Calculate Hz/sample and the indices corresponding to the noise cutoffs
    binfactor = dur_smp/fs
    LPbin = round(lco*binfactor)
    HPbin = round(hco*binfactor)

    # Synthesize normally distributed random complex numbers for all frequency bins in noise band
    a = np.zeros((dur_smp))
    b = np.zeros((dur_smp))
    a[LPbin:HPbin] = np.random.randn(HPbin-LPbin)
    b[LPbin:HPbin] = np.random.randn(HPbin-LPbin)
    fspec = a + b*1j

    # Calculate the frequencies of each FFT bin in kHz
    f_kHz = (np.arange(0, dur_smp)/binfactor)/1000

    # Define the signal-to-noise ratio at the output of auditory filters required to achieve detection threshold across
    # frequency range under power spectrum model (From personal correspondence with B. C. J. Moore via A. J. Oxenham)
    K = np.array([[0.0500, 13.5000],
                  [0.0630, 10.0000],
                  [0.0800, 7.2000],
                  [0.1000, 4.9000],
                  [0.1250, 3.1000],
                  [0.1600, 1.6000],
                  [0.2000, 0.4000],
                  [0.2500, -0.4000],
                  [0.3150, -1.2000],
                  [0.4000, -1.8500],
                  [0.5000, -2.4000],
                  [0.6300, -2.7000],
                  [0.7500, -2.8500],
                  [0.8000, -2.9000],
                  [1.0000, -3.0000],
                  [1.1000, -3.0000],
                  [2.0000, -3.0000],
                  [4.0000, -3.0000],
                  [8.0000, -3.0000],
                  [10.0000, -3.0000],
                  [15.0000, -3.0000]])

    # Interpolate K at all frequencies in noise band
    K_interp = interp1d(x=K[:, 0], y=K[:, 1], kind="slinear", fill_value="extrapolate")
    KdB = K_interp(f_kHz)

    # Calculate ERB at each freq
    ERB = 24.7 * ((4.37 * f_kHz) + 1)

    # Calculate P_s/N_o = K*ERB, i.e., ratio of signal power at threshold to noise power spectrum level
    cr_erb = KdB + (10*np.log10(ERB))
    TEN_No = -cr_erb

    # Identify indices included in ERB centered at 1 kHz
    idx_1kHz_ERB = np.logical_and(0.935 < f_kHz, f_kHz < 1.0681)

    # Create copy of noise and then filter it
    fspecfilt = deepcopy(fspec)
    fspecfilt[LPbin:HPbin] = fspecfilt[LPbin:HPbin] * 10**(TEN_No[LPbin:HPbin]/20)

    # Calculate rms in the 1 kHz ERB
    rms_1kHz_ERB = np.sqrt(np.sum(np.abs(fspecfilt[idx_1kHz_ERB])**2) / dur_smp**2)
    # Convert to level in dB SPL
    level_1kHz_ERB = 20*np.log10(rms_1kHz_ERB/20e-6)

    # Take real part of ifft to synthesize the noise
    noise = np.fft.ifft(fspecfilt)
    noise = np.real(noise[0:dur_smp])

    # Scale the noise such that the new level in the 1 kHz ERB matches the requested level
    noise = amplify(noise, level-level_1kHz_ERB+3)  # scale by 3 dB b/c we only synthesized LHS spectrum
    return noise
