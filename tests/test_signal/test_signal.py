import apcmodels.signal as sg
import numpy as np


def test_amplify_in_place():
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


def test_complex_tone_error():
    """ Check that complex_tone correctly handles freq and level arrays of different sizes"""
    try:
        sg.complex_tone([1, 2, 3], [1, 2], [1, 2, 3], 1, int(48e3))
        return False
    except Exception:
        return True


def test_cosine_ramp_durations():
    """ Check that cosine_ramp applied to a complex_tone can synthesize at a range of durations and sampling rates"""
    sg.cosine_ramp(sg.complex_tone([1, 2, 3], [1, 2, 3], [1, 2, 3], 1, int(48e3)), 0.1, int(48e3))
    sg.cosine_ramp(sg.complex_tone([1, 2, 3], [1, 2, 3], [1, 2, 3], 1, int(48e3)), 0.19923, int(48e3))
    sg.cosine_ramp(sg.complex_tone([1, 2, 3], [1, 2, 3], [1, 2, 3], 1, int(48e3)), 0.1, int(92332))
    sg.cosine_ramp(sg.complex_tone([1, 2, 3], [1, 2, 3], [1, 2, 3], 1, int(48e3)), 0.19923, int(92332))


def test_dbspl_pure_tone():
    """ Check that pure tone with peak 1 calibrates to 90.9 dB SPL"""
    np.testing.assert_approx_equal(sg.dbspl_pascal(sg.pure_tone(1000, 0, 1, int(48e3))), 90.9, significant=1)


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



