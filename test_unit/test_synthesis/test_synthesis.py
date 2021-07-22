from apcmodels import synthesis as sy
import apcmodels.signal as sg
from apcmodels.simulation import *
import numpy as np
import pytest


def test_synthesizer_synthesize_parameter_sequence_bad_inputs():
    """ Check that synthesize_sequence() correctly raises an error if you give it bad input types """
    try:
        synth = sy.Synthesizer()
        results = synth.synthesize_sequence(1)
        raise Exception('This should have failed!')
    except TypeError:
        return


def test_synthesizer_synthesize_parameter_sequence():
    """ Check that synthesize_sequence() correctly accepts a list of dicts and returns a corresponding number
    of stimuli"""
    synth = sy.Synthesizer()
    results = synth.synthesize_sequence([{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}])
    assert len(results) == 2


def test_synthesizer_synthesize_parameter_sequence_parameters():
    """ Check that synthesize_sequence() correctly accepts a Parametrs object and returns a corresponding number
    of stimuli """
    synth = sy.Synthesizer()
    results = synth.synthesize_sequence(Parameters())
    assert len(results) == 1


def test_synthesizer_synthesize_parameter_sequence_with_kwarg():
    """ Check that synthesize_sequence() correctly accepts a list of dicts and returns a corresponding number
    of stimuli while also allowing the user to pass additional keyword arguments"""
    synth = sy.Synthesizer()
    results = synth.synthesize_sequence([{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}], qux='hello world')
    assert len(results) == 2 and results[0]['qux'] == 'hello world'


def test_synthesizer_synthesize_parameter_sequence_with_duplicated_kwarg():
    """ Check that synthesize_sequence() correctly accepts a list of dicts and returns a corresponding number
    of stimuli, but if the user passes a kwarg that is already passed once by the parameter sequence then an error is re
    turned"""
    synth = sy.Synthesizer()
    try:
        synth.synthesize_sequence([{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}], foo='hello world')
        raise Exception
    except TypeError:
        return


def test_synthesizer_synthesize_parameter_sequence_array():
    """ Check that synthesize_sequence() correctly accepts an array of dicts and returns a corresponding number
    of stimuli """
    synth = sy.Synthesizer()
    results = synth.synthesize_sequence(np.array([{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}]))
    assert len(results) == 2


def test_synthesizer_synthesize_parameter_sequence_array_with_kwarg():
    """ Check that synthesize_sequence() correctly accepts an array of dicts and returns a corresponding number
    of stimuli and if we pass an extra kwarg that it gets passed correctly """
    synth = sy.Synthesizer()
    results = synth.synthesize_sequence(np.array([{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}]), qux='hello world')
    assert len(results) == 2 and results[0]['qux'] == 'hello world'


def test_synthesizer_synthesize_parameter_sequence_array_2d():
    """ Check that synthesize_sequence() correctly accepts a 2d array of dicts and returns a corresponding number
    of stimuli """
    synth = sy.Synthesizer()
    results = synth.synthesize_sequence(np.array([[{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}],
                                                  [{'foo': 10, 'bar': 20}, {'foo': 30, 'bar': 40}]]))
    assert results.shape == (2, 2)


def test_synthesizer_synthesize_parameter_sequence_array_nested():
    """ Check that synthesize_sequence() correctly accepts an array of a list of dicts and returns a corresponding number
    of stimuli """
    synth = sy.Synthesizer()
    params = np.array([{'a': 1}])
    params = increment_parameters(params, {'a': 0.01})
    results = synth.synthesize_sequence(params)
    
    assert results.shape == (1,) and type(results[0]) == list


def test_synthesizer_synthesize_parameter_sequence_nested():
    """ Check that synthesize_sequence() correctly accepts a lists of lists and returns a properly nested
    list of lists """
    synth = sy.Synthesizer()
    results = synth.synthesize_sequence([[{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}],
                                         [{'foo': 1, 'bar': 2}, {'foo': -3, 'bar': -4}, {'foo': 0, 'bar': 0}]])
    assert len(results) == 2 and len(results[0]) == 2 and len(results[1]) == 3


def test_flatten_parameters():
    """ Check that the flatten method correctly flattens arbitrary lists """
    assert len(flatten_parameters([1, 2, 3])) == 3 and \
           len(flatten_parameters([1, [2, 3], [4, 5], [[6, 7, 8]]])) == 8


def test_synthesizer_synthesize():
    """ Check that Synthesizer object can successfully synthesize"""
    synth = sy.Synthesizer()
    synth.synthesize()


def test_puretone_synthesize():
    """ Check that PureTone object can successfully synthesize and replicates standard pure tone synthesis"""
    synth = sy.PureTone()
    output = synth.synthesize(1000, 50, 0, 1, 0.1, int(48e3))
    reference = sg.cosine_ramp(sg.scale_dbspl(sg.pure_tone(1000, 0, 1, int(48e3)), 50), 0.1, int(48e3))
    assert np.all(output == reference)


def test_complextone_synthesize():
    """ Check that ComplexTone object can successfully synthesize and replicates standard complex tone synthesis"""
    synth = sy.ComplexTone()
    output = synth.synthesize(100, 1, 10, 50, 0, 1, 0.1, int(48e3))
    reference = sg.cosine_ramp(sg.complex_tone(100*np.arange(1, 11), 50*np.ones(10), np.zeros(10), 1, int(48e3)), 0.1, int(48e3))
    assert np.all(output == reference)


def test_puretone_run_sequence():
    """ Check that puretone's synthesize_sequence() correctly accepts a list of dicts and returns a correspond
    ing number of stimuli """
    synth = sy.PureTone()
    # Generate 1000 and 2000 Hz pure tones using synthesize_parameter_sequence
    results = synth.synthesize_sequence(
        [{'freq': 1000, 'level': 50, 'phase': 0, 'dur': 1, 'fs': int(48e3), 'dur_ramp': 0.1},
         {'freq': 2000, 'level': 50, 'phase': 0, 'dur': 1, 'fs': int(48e3), 'dur_ramp': 0.1}])
    # Generate references manually
    reference1 = sg.cosine_ramp(sg.scale_dbspl(sg.pure_tone(1000, 0, 1, int(48e3)), 50), 0.1, int(48e3))
    reference2 = sg.cosine_ramp(sg.scale_dbspl(sg.pure_tone(2000, 0, 1, int(48e3)), 50), 0.1, int(48e3))
    assert np.all(results[0] == reference1) and np.all(results[1] == reference2)


def test_puretone_incremented_sequence():
    """ Check that puretone's synthesize_sequence() and increment_sequence() successfully combine to
    produce two pure tones, one with a slightly higher frequency"""
    synth = sy.PureTone()
    # Generate 1000 and 2000 Hz pure tones using synthesize_parameter_sequence
    results = synth.synthesize_sequence(
        increment_parameters(parameters={'freq': 1000}, increments={'freq': 0.001}))
    # Generate references manually
    reference1 = sg.cosine_ramp(sg.scale_dbspl(sg.pure_tone(1000, 0, 1, int(48e3)), 50), 0.1, int(48e3))
    reference2 = sg.cosine_ramp(sg.scale_dbspl(sg.pure_tone(1000.001, 0, 1, int(48e3)), 50), 0.1, int(48e3))
    assert np.all(results[0] == reference1) and np.all(results[1] == reference2)


def test_puretone_random_level():
    """ Check that pure tone can accept a random level as a parameter and handle its evaluation when synthesis() is
    called. We check this by making sure that the output RMS level is not the same from one sample to another. """
    synth = sy.PureTone()
    tempfunc = lambda: np.random.uniform(40, 60, 1)
    assert sg.rms(synth.synthesize_sequence(parameters=[{'level': tempfunc}])[0]) != \
           sg.rms(synth.synthesize_sequence(parameters=[{'level': tempfunc}])[0])


def test_puretone_incremented_level():
    """ Check that pure tone can accept a level with an increment and return appropriately scaled pure tones """
    synth = sy.PureTone()
    params = increment_parameters({'level': 20}, {'level': 1})
    np.testing.assert_approx_equal(sg.dbspl_pascal(synth.synthesize_sequence(parameters=params)[0]) -
                                   sg.dbspl_pascal(synth.synthesize_sequence(parameters=params)[1]),
                                   -1, 5)


def test_puretone_incremented_level_with_random_level():
    """ Check that pure tone can accept a random level as a parameter with an increment and return appropriately scaled
     pure tones. We simplify the calculation by using a ~random~ distribution with no variance. """
    synth = sy.PureTone()
    tempfunc = lambda: np.random.uniform(50, 50, 1)
    params = increment_parameters({'level': tempfunc}, {'level': 1})
    np.testing.assert_approx_equal(sg.dbspl_pascal(synth.synthesize_sequence(parameters=params)[0]) -
                                   sg.dbspl_pascal(synth.synthesize_sequence(parameters=params)[1]),
                                   -1, 5)


def test_puretone_wiggled_level_with_random_variables():
    """ Check that we can construct a parameter dict, wiggle level to be various random variables, and then
    synthesize and get plausible output values. """
    synth = sy.PureTone()
    params = wiggle_parameters(dict(), 'level', [lambda: np.random.uniform(35, 45, 1),
                                                 lambda: np.random.uniform(45, 55, 1)])
    outs = synth.synthesize_sequence(parameters=params)
    assert sg.rms(outs[0]) < sg.rms(outs[1])


def test_synthesizer_raises_warnings_about_kwargs():
    """ Check that when we pass through kwargs to Synthesizer objects that we do get warnings """
    synth = sy.PureTone()
    with pytest.warns(UserWarning):
        synth.synthesize(freq=1000, testparam='foo')
