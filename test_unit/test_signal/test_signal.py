import apcmodels.signal as sg
import numpy as np


def test_amplify_by_zero_db():
    """ Check that rms does not change when amplifying a signal by 0 dB"""
    baseline_signal = sg.pure_tone(1000, 0, 1, int(48e3))
    amplified_signal = sg.amplify(baseline_signal, 0)
    assert sg.rms(baseline_signal) == sg.rms(amplified_signal)


def test_amplify_power_and_amplitude():
    """ Check that amplifying a signal by 10 dB produces the corresponding changes in power and amplitude"""
    baseline_signal = sg.pure_tone(1000, 0, 1, int(48e3))
    amplified_signal = sg.amplify(baseline_signal, 6)
    # Test that amplitude ratio is about 2
    np.testing.assert_approx_equal(np.max(amplified_signal)/np.max(baseline_signal), 1.9995, significant=3)
    # Test that power ratio is about 4
    np.testing.assert_approx_equal(np.max(amplified_signal)**2/np.max(baseline_signal**2), 3.981, significant=3)


def test_amplify_power_and_amplitude_decrease():
    """ Check that amplifying a signal by -10 dB produces the corresponding changes in power and amplitude"""
    baseline_signal = sg.pure_tone(1000, 0, 1, int(48e3))
    amplified_signal = sg.amplify(baseline_signal, -6)
    # Test that amplitude ratio is about 2
    np.testing.assert_approx_equal(np.max(amplified_signal)/np.max(baseline_signal), 0.501, significant=3)
    # Test that power ratio is about 4
    np.testing.assert_approx_equal(np.max(amplified_signal)**2/np.max(baseline_signal**2), 0.251, significant=3)


def test_complex_tone_error():
    """ Check that complex_tone correctly throws an error if you provide freq and level arrays of different sizes"""
    try:
        sg.complex_tone(np.array([1, 2, 3]), np.array([1, 2]), np.array([1, 2, 3]), 1, int(48e3))
        return False
    except Exception:
        return True


def test_complex_tone_single_component():
    """ Check that complex_tone correctly synthesizes an individual pure tone component """
    assert sg.complex_tone(np.array([1]), np.array([1]), np.array([1]), 1, int(48e3)).shape == (48e3, )


def test_cosine_ramp_durations():
    """ Check that cosine_ramp applied to a complex_tone can synthesize at a range of durations and sampling rates"""
    sg.cosine_ramp(sg.complex_tone(np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3]), 1, int(48e3)), 0.1, int(48e3))
    sg.cosine_ramp(sg.complex_tone(np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3]), 1, int(48e3)), 0.19923, int(48e3))
    sg.cosine_ramp(sg.complex_tone(np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3]), 1, int(48e3)), 0.1, int(92332))
    sg.cosine_ramp(sg.complex_tone(np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3]), 1, int(48e3)), 0.19923, int(92332))


def test_ramp_error():
    """ Check that if we request a ramp that is longer than the stimulus we get an error """
    try:
        sg.cosine_ramp(np.zeros(500), 1, int(48e3))
        raise Exception('This should have failed!')
    except ValueError:
        return


def test_dbspl_pure_tone_peak1():
    """ Check that pure tone with peak 1 calibrates to 90.9 dB SPL"""
    np.testing.assert_approx_equal(sg.dbspl_pascal(sg.pure_tone(1000, 0, 1, int(48e3))), 90.9, significant=1)


def test_dbspl_pure_tone_rms1():
    """ Check that pure tone with RMS 1 calibrates to 90.9 dB SPL"""
    x = sg.pure_tone(1000, 0, 1, int(48e3))
    x = x/sg.rms(x)
    np.testing.assert_approx_equal(sg.dbspl_pascal(x), 100, significant=1)


def test_dbspl_error_empty_array():
    """ Check that an empty array raises an error """
    try:
        sg.dbspl_pascal(np.zeros(500))
        raise Exception('This should have failed!')
    except ValueError:
        return


def test_dbspl_error_one_channel_empty():
    """ Check that an array with one empty channel still raises an error """
    try:
        stim = np.stack([sg.pure_tone(1, 0, 1, int(48e3)), np.zeros(int(48e3))], axis=1)
        sg.dbspl_pascal(stim)
        raise Exception('This should have failed!')
    except ValueError:
        return


def test_pure_tone_single_frequency():
    """ Check to make sure that manually synthesizing a pure tone produces the same results as pure_tone() """
    fs = int(48e3)
    f = 1000
    t = np.linspace(0, 1, fs)
    target = np.sin(2*np.pi*t*f)
    output = sg.pure_tone(f, 0, 1, fs)
    assert np.max(np.abs(output-target)) < 1e-10


def test_pure_tone_am_single_frequency_am_at_zero():
    """ Check to make sure that synthesizing an AM pure tone with a mod depth of zero produces the same results
     as pure_tone() """
    fs = int(48e3)
    f = 1000
    t = np.linspace(0, 1, fs)
    target = np.sin(2*np.pi*t*f)
    output = sg.pure_tone_am(f, 0, 1, 0, 0, 1, fs)
    assert np.max(np.abs(output-target)) < 1e-10


def test_pure_tone_multiple_components():
    """ Check to make sure that manually synthesizing a pure tone produces the same results as pure_tone() when we
    pass multiple components """
    fs = int(48e3)
    f = 1000
    t = np.linspace(0, 1, fs)
    target = np.sin(2*np.pi*t*f)
    output = sg.pure_tone(np.array([100, 200, f]), np.array([0, 0, 0]), 1, fs)
    assert np.max(np.abs(output[:, 2]-target)) < 1e-10


def test_pure_tone_am_depth():
    """ Check to make sure that synthesizing an AM pure tone with a mod depth of 1 produces a maxval near 2 """
    output = sg.pure_tone_am(1000, 0, 1, 0, 1, 1, int(48e3))
    np.testing.assert_approx_equal(np.max(output), 2, significant=3)


def test_pure_tone_am_depth2():
    """ Check to make sure that synthesizing an AM pure tone with a mod depth of 0.5 produces a maxval near 1.5 """
    output = sg.pure_tone_am(1000, 0, 1, 0, 0.5, 1, int(48e3))
    np.testing.assert_approx_equal(np.max(output), 1.5, significant=3)


def test_rms_errors():
    """ Check that rms correctly handles an empty array"""
    try:
        sg.rms(np.array([0]))
        return False
    except ValueError:
        return True


def test_rms_pure_tone():
    """ Check that pure tone with peak 1 calibrates to 0.707 rms"""
    np.testing.assert_approx_equal(sg.rms(sg.pure_tone(1000, 0, 1, int(48e3))), 0.707, significant=3)


def test_scale_dbspl_pure_tone():
    """ Check that we can specify a dB SPL value for a pure tone and have the tone calibrate to that value """
    baseline_signal = sg.pure_tone(1000, 0, 1, int(48e3))
    scaled_signal = sg.scale_dbspl(baseline_signal, 43)
    np.testing.assert_approx_equal(sg.dbspl_pascal(scaled_signal), 43, significant=3)


def test_te_noise():
    """ Synthesize a sample of threshold equalizing noise and check that the level in the 1 kHz ERB is equal to the
    requested level """
    level = 50
    fs = int(48e3)
    x = sg.te_noise(0.5, int(fs), 0, fs/2, level)  # calculate TE noise
    f = np.linspace(0, fs, x.shape[0])  # calculate frequency bins
    X = np.fft.fft(x)  # Calculate FFT
    X_sub = X[np.logical_and(0.935 < f/1000, f/1000 < 1.0681)]  # extract only those bins that are in 1 kHz ERB
    rms = np.sqrt(np.sum(np.abs(X_sub)**2) / x.shape[0]**2)  # calculate rms from selected bins
    level = 20*np.log10(rms/20e-6) + 3  # calculate level and adjust for fact that we're only looking at LHS of spectrum

    # Check assertion
    np.testing.assert_approx_equal(level, 50, significant=3)


def te_noise_errorgen():
    """ Test that te_noise throws errors when you pass an lco below 0 Hz """
    try:
        sg.te_noise(0, int(48e3), -1, 100, 50)
        raise Exception('This should have failed')
    except ValueError:
        return


def te_noise_errorgen2():
    """ Test that te_noise throws errors when you pass an hco above fs/2 Hz """
    try:
        sg.te_noise(0, int(48e3), 0, 30e3, 50)
        raise Exception('This should have failed')
    except ValueError:
        return