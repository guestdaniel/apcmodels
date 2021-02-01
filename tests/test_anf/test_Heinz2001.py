from apcmodels import signal as sg
import numpy as np
from apcmodels.anf import calculate_heinz2001_firing_rate

def test_null_inputs():
    """ Test to make sure that if model receives no input an error is raised """
    try:
        calculate_heinz2001_firing_rate(cfs=np.array([500]))
        raise Exception('This should have failed')
    except TypeError:
        return


def test_bare_minimum_input():
    """ Test to make sure that if model receives empty short stimulus it behaves normally """
    calculate_heinz2001_firing_rate(input=np.zeros(5000), cfs=np.array([500]))


def test_spontaneous_firing_rate():
    """ Test to make sure that if model receives empty short stimulus it has correct spont rate """
    results = calculate_heinz2001_firing_rate(input=np.zeros(5000), cfs=np.array([500]))
    np.testing.assert_almost_equal(np.mean(results), 50, decimal=1)