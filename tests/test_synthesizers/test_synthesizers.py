from apcmodels import synthesis as sy
import apcmodels.signal as sg
from apcmodels.simulation import *
import numpy as np


def test_synthesizer_increment_parameters_input_dict():
    """ Check that increment_parameters accepts a dict as input """
    synth = sy.Synthesizer()
    increment_parameters(baselines={'a': 1, 'b': 2}, increments={'a': 0.01})


def test_synthesizer_increment_parameters_input_list():
    """ Check that increment_parameters accepts a list as input """
    synth = sy.Synthesizer()
    increment_parameters(baselines=[{'a': 1, 'b': 2}], increments={'a': 0.01})


def test_synthesizer_increment_parameters_input_nested_list():
    """ Check that increment_parameters accepts a nested list as input """
    synth = sy.Synthesizer()
    increment_parameters(baselines=[{'a': 1, 'b': 2},
                                          [[{'a': 1, 'b': 2}, {'a': 1, 'b': 2}], [{'a': 2, 'b': 40}]]],
                               increments={'a': 0.01})


def test_synthesizer_wiggle_parameters_input_dict():
    """ Check that wiggle_parameters accepts a dict as input """
    synth = sy.Synthesizer()
    wiggle_parameters(baselines={'a': 1, 'b': 2}, parameter='a', values=[1, 2, 3, 4])


def test_synthesizer_wiggle_parameters_input_list():
    """ Check that wiggle_parameters accepts a list as input """
    synth = sy.Synthesizer()
    wiggle_parameters(baselines=[{'a': 1, 'b': 2}], parameter='a', values=[1, 2, 3, 4])


def test_synthesizer_wiggle_parameters_input_nested_list():
    """ Check that wiggle_parameters accepts a nested list as input """
    synth = sy.Synthesizer()
    wiggle_parameters(baselines=[{'a': 1, 'b': 2},
                                          [[{'a': 1, 'b': 2}, {'a': 1, 'b': 2}], [{'a': 2, 'b': 40}]]],
                            parameter='a', values=[1, 2, 3, 4])


def test_synthesizer_wiggle_parameters_repeated():
    """ Check that increment_parameters accepts a nested list as input """
    synth = sy.Synthesizer()
    wiggle_parameters(baselines=wiggle_parameters(baselines={'a': 1, 'b': 2},
                                                  parameter='a',
                                                  values=[1, 2, 3, 4]),
                            parameter='b',
                            values=[10, 20])


def test_synthesizer_synthesize_parameter_sequence():
    """ Check that synthesize_sequence() correctly accepts a list of dicts and returns a corresponding number
    of stimuli"""
    synth = sy.Synthesizer()
    results = synth.synthesize_sequence([{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}])
    assert len(results) == 2


def test_synthesizer_synthesize_parameter_sequence_with_kwarg():
    """ Check that synthesize_sequence() correctly accepts a list of dicts and returns a corresponding number
    of stimuli while also allowing the user to pass additional keyword arguments"""
    synth = sy.Synthesizer()
    results = synth.synthesize_sequence([{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}], qux='hello world')
    assert len(results) == 2


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


def test_synthesizer_synthesize_parameter_sequence_nested():
    """ Check that synthesize_sequence() correctly accepts a lists of lists and returns a properly nested
    list of lists """
    synth = sy.Synthesizer()
    results = synth.synthesize_sequence([[{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}],
                                         [{'foo': 1, 'bar': 2}, {'foo': -3, 'bar': -4}, {'foo': 0, 'bar': 0}]])
    assert len(results) == 2 and len(results[0]) == 2 and len(results[1]) == 3


def test_synthesizer_flatten_parameters():
    """ Check that the flatten method correctly flattens arbitrary lists """
    synth = sy.Synthesizer()
    assert len(flatten_parameters([1, 2, 3])) == 3 and \
           len(flatten_parameters([1, [2, 3], [4, 5], [[6, 7, 8]]])) == 8


def test_synthesizer_synthesize():
    """ Check that Synthesizer object can successfully synthesize"""
    synth = sy.Synthesizer()
    synth.synthesize()


def test_puretone_synthesize():
    """ Check that PureTone object can successfully synthesize and replicates standard puretone synthesis"""
    synth = sy.PureTone()
    output = synth.synthesize(1000, 50, 0, 1, 0.1, int(48e3))
    reference = sg.cosine_ramp(sg.scale_dbspl(sg.pure_tone(1000, 0, 1, int(48e3)), 50), 0.1, int(48e3))
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
        increment_parameters(baselines={'freq': 1000}, increments={'freq': 0.001}))
    # Generate references manually
    reference1 = sg.cosine_ramp(sg.scale_dbspl(sg.pure_tone(1000, 0, 1, int(48e3)), 50), 0.1, int(48e3))
    reference2 = sg.cosine_ramp(sg.scale_dbspl(sg.pure_tone(1000.001, 0, 1, int(48e3)), 50), 0.1, int(48e3))
    assert np.all(results[0] == reference1) and np.all(results[1] == reference2)


def test_puretone_random_level():
    """ Check that pure tone can accept a random level as a parameter and handle its evaluation when synthesis() is
    called. We check this by making sure that the output RMS level is not the same from one sample to another. """
    synth = sy.PureTone()
    tempfunc = lambda: np.random.uniform(40, 60, 1)
    assert sg.rms(synth.synthesize_sequence(parameter_sequence=[{'level': tempfunc}])[0]) != \
           sg.rms(synth.synthesize_sequence(parameter_sequence=[{'level': tempfunc}])[0])


def test_puretone_incremented_level():
    """ Check that pure tone can accept a level with an increment and return appropriately scaled pure tones """
    synth = sy.PureTone()
    params = increment_parameters({'level': 20}, {'level': 1})
    np.testing.assert_approx_equal(sg.dbspl_pascal(synth.synthesize_sequence(parameter_sequence=params)[0]) -
                                   sg.dbspl_pascal(synth.synthesize_sequence(parameter_sequence=params)[1]),
                                   -1, 5)


def test_puretone_incremented_level_with_random_level():
    """ Check that pure tone can accept a random level as a parameter with an increment and return appropriately scaled
     pure tones. We simplify the calculation by using a ~random~ distribution with no variance. """
    synth = sy.PureTone()
    tempfunc = lambda: np.random.uniform(50, 50, 1)
    params = increment_parameters({'level': tempfunc}, {'level': 1})
    np.testing.assert_approx_equal(sg.dbspl_pascal(synth.synthesize_sequence(parameter_sequence=params)[0]) -
                                   sg.dbspl_pascal(synth.synthesize_sequence(parameter_sequence=params)[1]),
                                   -1, 5)