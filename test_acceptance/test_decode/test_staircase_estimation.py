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
    #n_interval = 2
    #scale = 1/n_interval
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
    staircase = decode_staircase_procedure(sim.simulate, myrule, syn, 'freq', 10, [5, 2.5, 1, 0.5, 0.25, 0.1],
                                           transform=transform, max_trials=5000, max_num_reversal=30,
                                           calculate_from_reversal=12)

    results = sim.run(params, runfunc=lambda x: [staircase(ele) for ele in x], parallel=False)
    np.testing.assert_almost_equal(np.mean(results[0]), desired=target, decimal=0)