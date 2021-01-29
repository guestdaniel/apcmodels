import apcmodels.simulation as sy
import numpy as np


def test_simulator_construct_batch_size():
    """ Check that construct batch accepts stimulus and parameter sequences of variable sizes and returns a list of the
    appropriate size"""
    sim = sy.Simulator()
    batch_zip = sim.construct_batch([np.zeros(5), np.ones(5), 2 * np.ones(5)],
                                    [{'freq': 1000}, {'freq': 2000}, {'freq': 5000}],
                                    [{'CF': 50, 'fs': 10000}, {'CF': 100, 'fs': 10000}], 'zip')
    batch_product = sim.construct_batch([np.zeros(5), np.ones(5), 2 * np.ones(5)],
                                        [{'freq': 1000}, {'freq': 2000}, {'freq': 5000}],
                                        [{'CF': 50, 'fs': 10000}, {'CF': 100, 'fs': 10000}], 'product')
    assert len(batch_zip) == 2 and len(batch_product) == 6


def test_simulator_construct_batch_error_handling():
    """ Check that construct batch correctly throws an error if sizes of inputs don't match """
    sim = sy.Simulator()
    try:
        temp = sim.construct_batch([np.zeros(5), np.ones(5), 2 * np.ones(5)], [{'freq': 1000}, {'freq': 2000}],
                                   [{'CF': 50, 'fs': 10000}, {'CF': 100, 'fs': 10000}], 'zip')
        raise Exception('Should have failed!')
    except AssertionError:
        return


def test_simulator_run_complex_batch():
    """ Check that run can handle arbitrary inputs """
    # Create dummy function that returns input as output --- thus, the output of run() should be identical to its input
    def dummy(input):
        return input
    # Initialize simulator object
    sim = sy.Simulator()
    # Create dummy input
    dummy_input = ['yo', 'ye', 'ya']
    output = sim.run(runfunc=dummy, batch=dummy_input, parallel=True)
    # Check that all inputs and outputs align
    for this_input, this_output in zip(dummy_input, output):
        assert(this_input == this_output)


def test_simulator_run():
    """ Check that run() correctly accepts a list of dicts and returns a corresponding number of responses"""
    sim = sy.Simulator()
    results = sim.run(sim.simulate, [{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}], parallel=False)
    assert len(results) == 2


def test_simulator_run_parallel():
    """ Check that run() correctly accepts a list of dicts and returns a corresponding number of responses"""
    sim = sy.Simulator()
    results = sim.run(sim.simulate, [{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}], parallel=True)
    assert len(results) == 2


def test_synthesizer_synthesize_parameter_sequence():
    """ Check that synthesize_parameter_sequence() correctly accepts a list of dicts and returns a corresponding number
    of stimuli"""
    synth = sy.Synthesizer()
    results = synth.synthesize_parameter_sequence([{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}])
    assert len(results) == 2


def test_synthesizer_synthesize_parameter_sequence_with_kwarg():
    """ Check that synthesize_parameter_sequence() correctly accepts a list of dicts and returns a corresponding number
    of stimuli while also allowing the user to pass additional keyword arguments"""
    synth = sy.Synthesizer()
    results = synth.synthesize_parameter_sequence([{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}], qux='hello world')
    assert len(results) == 2


def test_synthesizer_synthesize_parameter_sequence_with_duplicated_kwarg():
    """ Check that synthesize_parameter_sequence() correctly accepts a list of dicts and returns a corresponding number
    of stimuli, but if the user passes a kwarg that is already passed once by the parameter sequence then an error is returned"""
    synth = sy.Synthesizer()
    try:
        synth.synthesize_parameter_sequence([{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}], foo='hello world')
        raise Exception
    except TypeError:
        return

def test_synthesizer_synthesize_parameter_sequence_nested():
    """ Check that synthesize_parameter_sequence() correctly accepts a lists of lists and returns a properly nested
    list of lists """
    synth = sy.Synthesizer()
    results = synth.synthesize_parameter_sequence([[{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}],
                                                   [{'foo': 1, 'bar': 2}, {'foo': -3, 'bar': -4}, {'foo': 0, 'bar': 0}]])
    assert len(results) == 2 and len(results[0]) == 2 and len(results[1]) == 3


def test_synthesizer_synthesize():
    """ Check that Synthesizer object can successfully synthesize"""
    synth = sy.Synthesizer()
    synth.synthesize()


def test_simulate_firing_rates_error_gen():
    """ Check that simulate firing rates raises exceptions if no input or model is provided """
    try:
        sy.simulate_firing_rates()
        raise Exception('Should have failed')
    except TypeError:
        return


