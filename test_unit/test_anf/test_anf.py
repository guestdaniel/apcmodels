# TODO: add functions to test calculate_auditory_nerve_response
# noticed weird behavior if we pass cf_low and n_cfs instead of n_cf ... investigate!
import apcmodels.anf as anf
import numpy as np
import pytest
import warnings


@pytest.mark.parametrize('anf_model', [anf.AuditoryNerveHeinz2001Numba, anf.AuditoryNerveZilany2014])
def test_anf_simulator(anf_model):
    """ Test to make sure that each anf Simulator object can be set up and run on a single input and does not raise any
    errors """
    # Initialize simulator object
    sim = anf_model()
    # Create dummy input
    dummy_input = [{'_input': np.zeros(5000), 'cf_low': 200, 'cf_high': 20000, 'n_cf': 20}]
    output = sim.run(params=dummy_input)


@pytest.mark.parametrize('anf_model', [anf.AuditoryNerveHeinz2001Numba, anf.AuditoryNerveZilany2014])
def test_anf_check_no_output_channels_simulator(anf_model):
    """ Test to make sure that a basic anf simulation can be set up and run on a single input and that the output
    has the right number of output channels """
    # Initialize simulator object
    sim = anf_model()
    # Create dummy input
    dummy_input = [{'_input': np.zeros(5000), 'cf_low': 200, 'cf_high': 20000, 'n_cf': 20}]
    output = sim.run(params=dummy_input)
    assert output[0].shape[0] == 20


@pytest.mark.parametrize('anf_model', [anf.AuditoryNerveHeinz2001Numba, anf.AuditoryNerveZilany2014])
def test_check_params_simulator(anf_model):
    """ Test to make sure that warnings are generated if models' default runfuncs see unexpected parameters """
    with pytest.warns(UserWarning):
        warnings.simplefilter('default')
        params = [{'_input': np.zeros(5000), 'cfs': np.array([500]), 'test': True}]
        sim = anf_model()
        sim.run(params, parallel=False)


@pytest.mark.parametrize('anf_func', [anf.calculate_zilany2014_firing_rate, anf.calculate_zilany2014_firing_rate])
def test_check_zilany2014_warnings(anf_func):
    """ Test to make sure that Zilany model raises warnings for weird CFs """
    with pytest.warns(UserWarning):
        warnings.simplefilter('default')
        params = {'_input': np.zeros(5000), 'fs': int(100e3), 'cfs': np.array([0])}
        anf_func(**params)


@pytest.mark.parametrize('anf_func', [anf.calculate_heinz2001_firing_rate, anf.calculate_zilany2014_firing_rate])
def test_null_inputs_rates(anf_func):
    """ Test to make sure that if model receives no input an error is raised """
    try:
        anf_func()
        raise Exception('This should have failed')
    except KeyError:
        return


@pytest.mark.parametrize('anf_func', [anf.calculate_heinz2001_firing_rate, anf.calculate_zilany2014_firing_rate])
def test_bare_minimum_input_rates(anf_func):
    """ Test to make sure that if model receives empty short stimulus it behaves normally """
    anf_func(_input=np.zeros(5000), cfs=np.array([500]))


@pytest.mark.parametrize(['anf_func', 'spont_rate'], [(anf.calculate_heinz2001_firing_rate, 50),
                                                      (anf.calculate_zilany2014_firing_rate, 100)])
def test_spontaneous_firing_rate_rates(anf_func, spont_rate):
    """ Test to make sure that if model receives empty short stimulus it has correct spont rate """
    results = anf_func(_input=np.zeros(10000), cfs=np.array([500]))
    np.testing.assert_almost_equal(np.mean(results), spont_rate, decimal=-1)


@pytest.mark.parametrize('anf_func', [anf.calculate_heinz2001_firing_rate, anf.calculate_zilany2014_firing_rate])
def test_output_dimensionality_rates(anf_func):
    """ Test to make sure that if model simulates multiple channels that they output with the correct shape """
    results = anf_func(_input=np.zeros(4000), cfs=np.array([1000, 2000, 3000]))
    assert results.shape == (3, 13000)


@pytest.mark.parametrize('anf_func', [anf.calculate_zilany2014_firing_rate])
def test_spontaneous_firing_rate_versus_fiber_type(anf_func):
    """ Test to make sure that the firing rate decreases for msr and lsr fibers over hsr fibers """
    rate_hsr = np.mean(
        anf_func(_input=np.zeros(50000), cfs=np.array([1000]), fiber_type='hsr'))
    rate_msr = np.mean(
        anf_func(_input=np.zeros(50000), cfs=np.array([1000]), fiber_type='msr'))
    rate_lsr = np.mean(
        anf_func(_input=np.zeros(50000), cfs=np.array([1000]), fiber_type='lsr'))

    assert np.all(np.diff([rate_hsr, rate_msr, rate_lsr]) < 0)


@pytest.mark.parametrize('anf_func', [anf.calculate_zilany2014_spikes])
def test_spontaneous_firing_rate_versus_fiber_type_spikes(anf_func):
    """ Test to make sure that the firing rate decreases for msr and lsr fibers over hsr fibers for the spiking
    simulator"""
    num_spike_hsr = len(anf_func(_input=np.zeros(100000), cfs=np.array([1000]), anf_num=(1, 0, 0),
                                                    fs=int(100e3)).iloc[0]['spikes'])
    num_spike_msr = len(anf_func(_input=np.zeros(100000), cfs=np.array([1000]), anf_num=(0, 1, 0),
                                                    fs=int(100e3)).iloc[0]['spikes'])
    num_spike_lsr = len(anf_func(_input=np.zeros(100000), cfs=np.array([1000]), anf_num=(0, 0, 1),
                                                    fs=int(100e3)).iloc[0]['spikes'])

    assert np.all(np.diff([num_spike_hsr, num_spike_msr, num_spike_lsr]))