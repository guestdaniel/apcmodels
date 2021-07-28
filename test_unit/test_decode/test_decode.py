from apcmodels.decode import *
import apcmodels.anf as anf
import apcmodels.synthesis as sy
import apcmodels.simulation as si
import numpy as np
from apcmodels.decode import decode_staircase_procedure
from random import random
import pytest


def logistic_psychometric_function(x, threshold, slope, scale):
    return scale/(1 + np.exp(-slope*(x-threshold))) + (1-scale)


def inverse_logistic_psychometric_function(y, threshold, slope, scale):
    return np.log(scale/(y-1+scale)-1) / (-slope) + threshold


def test_run_rates_util():
    """ Test to make sure that run_rates_util will correctly accept either a single input or a list of inputs and return
    the right output """
    def dummy_ratefunc(params):
        return params['_input']

    # Test to make sure that if you just provide it with single input it handles that okay
    output1 = run_rates_util(dummy_ratefunc, {'_input': 1})
    assert output1 == 1

    # Also test that it handles a list and returns the input
    output2 = run_rates_util(dummy_ratefunc, [{'_input': 1}, {'_input': 2}, {'_input': 3}])
    for _in, _out in zip([1, 2, 3], output2):
        assert _in == _out


def test_ideal_observer_single_input():
    """ Test that if we provide a single stimulus to a ratefunc wrapped in decode_ideal_observer that some sort of
     error is raised to indicate that an ideal observer can't be calculated based on a single simulation! """
    # Initialize simulator object
    sim = anf.AuditoryNerveHeinz2001()

    # Define stimulus parameters
    fs = int(200e3)
    tone_level = 30
    tone_dur = 0.1
    tone_ramp_dur = 0.01
    tone_freq = 1000

    # Synthesize stimuli
    synth = sy.PureTone()
    params = {'level': tone_level, 'dur': tone_dur, 'dur_ramp': tone_ramp_dur, 'freq': tone_freq, 'fs': fs}
    stimuli = synth.synthesize_sequence([params])

    # Add stimuli and model params
    params = si.append_parameters(params, ['_input', 'cf_low', 'cf_high', 'n_cf'], [stimuli[0], 1000, 1000, 1])
    params = si.append_parameters(params, ['n_fiber_per_chan', 'fs', 'delta_theta', 'API'],
                                  [5, int(200e3), [0.001], np.zeros(1)])

    # Run model
    try:
        sim.run([params], runfunc=decode_ideal_observer(sim.simulate))
        raise Exception('This should have failed!')
    except ValueError:
        return


def test_ideal_observer_valid_input():
    """ Test that if we provide a valid params to a ratefunc wrapped in decode_ideal_observer that everything runs
     without any errors """
    # Initialize simulator object
    sim = anf.AuditoryNerveHeinz2001()

    # Define stimulus parameters
    fs = int(200e3)
    tone_level = 30
    tone_dur = 0.1
    tone_ramp_dur = 0.01
    tone_freq = 1000

    # Synthesize stimuli
    synth = sy.PureTone()
    params = si.Parameters(level=tone_level, dur=tone_dur, dur_ramp=tone_ramp_dur, freq=tone_freq, fs=fs)
    params.increment({'freq': 0.001})
    stimuli = synth.synthesize_sequence(params)
    params.add_inputs(stimuli)

    # Add stimuli and model params
    params.append(['cf_low', 'cf_high', 'n_cf'], [1000, 1000, 1])
    params.append(['n_fiber_per_chan', 'fs', 'delta_theta', 'API'], [1, int(200e3), [0.001], np.zeros(1)])

    # Run
    out = sim.run(params, runfunc=decode_ideal_observer(sim.simulate))


@pytest.mark.parametrize('_type', ['AI', 'RP'])
def test_compute_partial_derivative_matrix_calculations(_type):
    """ Test that if we pass sensible outputs to the partial derivative calculator that the outputs are correct """
    x = [np.ones((1, 5000)), 1 + np.ones((1, 5000))]
    fs = int(20e3)
    delta_theta = [1]
    n_fiber_per_chan = 1
    np.testing.assert_almost_equal(compute_partial_derivative_matrix(x, fs, delta_theta, n_fiber_per_chan, _type)[0, 0],
                                   0.125, decimal=5)


@pytest.mark.parametrize('n_chan', [1, 5])
@pytest.mark.parametrize('n_param', [2, 5])
@pytest.mark.parametrize('_type', ['AI', 'RP'])
def test_compute_partial_derivative_matrix_shape(n_chan, n_param, _type):
    """ Test that if we pass sensible inputs to the partial derivative calculator that the outputs shaped right """
    x = list()
    for param in range(n_param):
        x.append(np.zeros((n_chan, 5000)) + param)
    delta_theta = [1]*(n_param-1)
    fs = int(20e3)
    n_fiber_per_chan = 1
    assert compute_partial_derivative_matrix(x, fs, delta_theta, n_fiber_per_chan, _type).shape == \
           (n_param-1, n_param-1)


def test_calculate_threshold_input_to_output_shape():
    """ Test that calculate_threshold correctly handles different sizes of input matrices """
    # Scalar input
    assert calculate_threshold(np.array(1), np.array(1)).shape == ()
    # Matrix input
    assert calculate_threshold(np.array([[1, 0], [0, 1]]), np.array([[1, 1], [1, 1]])).shape == ()
    # Stacked matrix input
    assert calculate_threshold(np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]]), np.array([[1, 1], [1, 1]])).shape == ()


def test_find_parameter():
    """ Test to make sure that find_parameter accepts a nested list of params and returns the appropriate value """
    params = [[{'a': 5, 'b': 3}], [{'c': 2}]]
    assert find_parameter(params, 'c') == 2


@pytest.mark.parametrize('threshold', [-15, -10, -5, 0, 5, 10, 15])
def test_staircase_estimation_for_different_thresholds(threshold):
    """ Estimate arbitrary thresholds using staircase procedure with a logistic psychometric function """

    # Define new subclass of Simulator that provides us with a dummy output (simply returning _input as output)
    class DummyModel(si.Simulator):
        def __init__(self):
            super().__init__()

        def simulate(self, params, **kwargs):
            return kwargs['_input']

    # Define a new subclass of Synthesizer that provides us with a dummy output (simply returning freq as output)
    class DummyInput(sy.Synthesizer):
        def __init__(self):
            super().__init__(stimulus_name='Dummy')

        def synthesize(self, **kwargs):
            return kwargs['freq']

    # Select simulator and synthesizer
    sim = DummyModel()
    syn = DummyInput()

    # Set up parameters
    params = si.Parameters(freq=100)
    params.repeat(10)

    # Define transform between 10*log10(%) and delta_f
    def transform(x, params):
        return params['freq'] * (10 ** (x / 10)) / 100

    # Define parameters for psychometric function
    slope = 1
    p = 0.707  # 2-down-1-up procedure should track 0.707 point
    scale = 1
    target = inverse_logistic_psychometric_function(p, threshold, slope, scale)

    # Define rule for selecting which simulation response contains the target
    def myrule(sims, params):
        # Look at each simulation and compute the 10*log10(%) --- we can do this because we're using dummy sims/models
        decision_variable = 10*np.log10((sims[1] - sims[0])/sims[0] * 100)  # 10*log10(%)
        # Now, respond based on a sigmoid psychometric function
        if logistic_psychometric_function(decision_variable, threshold, slope, scale) > random():
            return 2
        else:
            return 1

    # Define staircase function
    staircase = decode_staircase_procedure(sim.simulate, myrule, syn, 'freq', threshold+5, [5, 2.5, 1, 0.5, 0.25],
                                           transform=transform, max_trials=50000, max_num_reversal=50,
                                           calculate_from_reversal=20)

    results = sim.run(params, runfunc=lambda x: [staircase(ele) for ele in x], parallel=False)
    np.testing.assert_almost_equal(np.mean(results[0]), desired=target, decimal=0)


@pytest.mark.parametrize(['p', 'n_down', 'n_up'],
                         [(0.159, 1, 4), (0.293, 1, 2), (0.5, 1, 1), (0.707, 2, 1), (0.794, 3, 1), (0.841, 4, 1)])
def test_staircase_estimation_for_different_thresholds(p, n_down, n_up):
    """ Estimate various points along psychometric function using staircase procedure with a logistic psychometric
    function """

    # Define new subclass of Simulator that provides us with a dummy output (simply returning _input as output)
    class DummyModel(si.Simulator):
        def __init__(self):
            super().__init__()

        def simulate(self, params, **kwargs):
            return kwargs['_input']

    # Define a new subclass of Synthesizer that provides us with a dummy output (simply returning freq as output)
    class DummyInput(sy.Synthesizer):
        def __init__(self):
            super().__init__(stimulus_name='Dummy')

        def synthesize(self, **kwargs):
            return kwargs['freq']

    # Select simulator and synthesizer
    sim = DummyModel()
    syn = DummyInput()

    # Set up parameters
    params = si.Parameters(freq=100)
    params.repeat(10)

    # Define transform between 10*log10(%) and delta_f
    def transform(x, params):
        return params['freq'] * (10 ** (x / 10)) / 100

    # Define parameters for psychometric function
    threshold = 5
    slope = 1
    scale = 1
    target = inverse_logistic_psychometric_function(p, threshold, slope, scale)

    # Define rule for selecting which simulation response contains the target
    def myrule(sims, params):
        # Look at each simulation and compute the 10*log10(%) --- we can do this because we're using dummy sims/models
        decision_variable = 10*np.log10((sims[1] - sims[0])/sims[0] * 100)  # 10*log10(%)
        # Now, respond based on a sigmoid psychometric function
        if logistic_psychometric_function(decision_variable, threshold, slope, scale) > random():
            return 2
        else:
            return 1

    # Define staircase function
    staircase = decode_staircase_procedure(sim.simulate, myrule, syn, 'freq', threshold+5, [5, 2.5, 1, 0.5, 0.25],
                                           n_down=n_down, n_up=n_up, transform=transform, max_trials=50000,
                                           max_num_reversal=50, calculate_from_reversal=20)

    results = sim.run(params, runfunc=lambda x: [staircase(ele) for ele in x], parallel=False)
    np.testing.assert_almost_equal(np.mean(results[0]), desired=target, decimal=0)


def test_staircase_estimation_max_value():
    """ Verify that we fail to estimate a threshold that exceeds our maximum value """

    # Define new subclass of Simulator that provides us with a dummy output (simply returning _input as output)
    class DummyModel(si.Simulator):
        def __init__(self):
            super().__init__()

        def simulate(self, params, **kwargs):
            return kwargs['_input']

    # Define a new subclass of Synthesizer that provides us with a dummy output (simply returning freq as output)
    class DummyInput(sy.Synthesizer):
        def __init__(self):
            super().__init__(stimulus_name='Dummy')

        def synthesize(self, **kwargs):
            return kwargs['freq']

    # Select simulator and synthesizer
    sim = DummyModel()
    syn = DummyInput()

    # Set up parameters
    params = si.Parameters(freq=100)
    params.repeat(10)

    # Define transform between 10*log10(%) and delta_f
    def transform(x, params):
        return params['freq'] * (10 ** (x / 10)) / 100

    # Define parameters for psychometric function
    threshold = 35
    slope = 1
    scale = 1
    p = 0.707

    # Define rule for selecting which simulation response contains the target
    def myrule(sims, params):
        # Look at each simulation and compute the 10*log10(%) --- we can do this because we're using dummy sims/models
        decision_variable = 10*np.log10((sims[1] - sims[0])/sims[0] * 100)  # 10*log10(%)
        # Now, respond based on a sigmoid psychometric function
        if logistic_psychometric_function(decision_variable, threshold, slope, scale) > random():
            return 2
        else:
            return 1

    # Define staircase function
    staircase = decode_staircase_procedure(sim.simulate, myrule, syn, 'freq', threshold+5, [5, 2.5, 1, 0.5, 0.25],
                                           transform=transform, max_trials=50000, max_num_reversal=12, max_value=20,
                                           calculate_from_reversal=6)

    results = sim.run(params, runfunc=lambda x: [staircase(ele) for ele in x], parallel=False)
    assert np.isnan(np.mean(results[0]))
