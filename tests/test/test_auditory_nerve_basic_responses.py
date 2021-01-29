## SKETCHES
import apcmodels.anf as anf
import apcmodels.simulation as sy
import numpy as np

def test_anf():
    """ Test to make sure that a basic anf simulation can be set up and run on a single input"""
    # Initialize simulator object
    sim = sy.Simulator()
    # Create dummy input
    dummy_input = [{'input': np.zeros(5000), 'cf_low': 200, 'cf_high': 20000, 'n_cf': 20}]
    output = sim.run(runfunc=lambda paramdict: sy.simulate_firing_rates(anf.calculate_Heinz2001_firing_rate, paramdict),
                     batch=dummy_input, parallel=True)


def test_anf_check_no_output_channels():
    """ Test to make sure that a basic anf simulation can be set up and run on a single input and that the output
    has the right number of output channels """
    # Initialize simulator object
    sim = sy.Simulator()
    # Create dummy input
    dummy_input = [{'input': np.zeros(5000), 'cf_low': 200, 'cf_high': 20000, 'n_cf': 20}]
    output = sim.run(runfunc=lambda paramdict: sy.simulate_firing_rates(anf.calculate_Heinz2001_firing_rate, paramdict),
                     batch=dummy_input, parallel=True)
    assert output[0].shape[0] == 20


def test_anf_rate_level_function():
    """ Test to make sure that a basic anf simulation can be set up and run on a single input and that the output
    has the right number of output channels """
    # Initialize simulator object
    sim = sy.Simulator()
    # Create dummy input
    dummy_input = [{'input': np.zeros(5000), 'cf_low': 200, 'cf_high': 20000, 'n_cf': 20}]
    output = sim.run(runfunc=lambda paramdict: sy.simulate_firing_rates(anf.calculate_Heinz2001_firing_rate, paramdict),
                     batch=dummy_input, parallel=True)
    assert output[0].shape[0] == 20