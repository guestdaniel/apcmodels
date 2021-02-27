# TODO: add functions to test calculate_auditory_nerve_firing_rate
# noticed weird behavior if we pass cf_low and n_cfs instead of n_cf ... investigate!
import apcmodels.anf as anf
import numpy as np


def test_anf():
    """ Test to make sure that a basic anf simulation can be set up and run on a single input"""
    # Initialize simulator object
    sim = anf.AuditoryNerveHeinz2001Numba()
    # Create dummy input
    dummy_input = [{'_input': np.zeros(5000), 'cf_low': 200, 'cf_high': 20000, 'n_cf': 20}]
    output = sim.run(params=dummy_input,
                     parallel=True)


def test_anf_check_no_output_channels():
    """ Test to make sure that a basic anf simulation can be set up and run on a single input and that the output
    has the right number of output channels """
    # Initialize simulator object
    sim = anf.AuditoryNerveHeinz2001Numba()
    # Create dummy input
    dummy_input = [{'_input': np.zeros(5000), 'cf_low': 200, 'cf_high': 20000, 'n_cf': 20}]
    output = sim.run(params=dummy_input,
                     parallel=True)
    assert output[0].shape[0] == 20