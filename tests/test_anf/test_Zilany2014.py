from apcmodels import signal as sg
import numpy as np
from apcmodels.anf import calculate_zilany2014_firing_rate


def test_null_inputs():
    """ Test to make sure that if model receives no input an error is raised """
    try:
        calculate_zilany2014_firing_rate()
        raise Exception('This should have failed')
    except KeyError:
        return


def test_bare_minimum_input():
    """ Test to make sure that if model receives empty short stimulus it behaves normally """
    calculate_zilany2014_firing_rate(_input=np.zeros(5000), cfs=np.array([500]))


def test_spontaneous_firing_rate():
    """ Test to make sure that if model receives empty short stimulus it has correct spont rate """
    results = calculate_zilany2014_firing_rate(_input=np.zeros(50000), cfs=np.array([1000]))
    np.testing.assert_almost_equal(np.mean(results), 100, decimal=-1)  # check we're within 10 sp/s of 100 sp/s


def test_output_dimensionality():
    """ Test to make sure that if model simulates multiple channels that they output with the correct shape """
    results = calculate_zilany2014_firing_rate(_input=np.zeros(4000), cfs=np.array([1000, 2000, 3000]))
    assert results.shape == (3, 8000)