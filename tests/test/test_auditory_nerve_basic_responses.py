import apcmodels.simulation as si
import apcmodels.synthesis as sy
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


def test_anf_rate_level_function():
    """ Test to make sure that a basic anf simulation can be set up and run on a single sequence of pure tones with
     increasing levels and that the simulation returns a corresponding increasing rate response """
    # Initialize simulator object
    sim = anf.AuditoryNerveHeinz2001Numba()

    # Define stimulus parameters
    fs = int(200e3)
    tone_freq = 1000
    tone_dur = 0.1
    tone_ramp_dur = 0.01
    tone_levels = [0, 10, 20, 30, 40, 50]

    # Synthesize stimuli
    synth = sy.PureTone()
    params = {'freq': tone_freq, 'dur': tone_dur, 'dur_ramp': tone_ramp_dur, 'fs': fs}
    params = si.wiggle_parameters(params, parameter_to_wiggle='level', values=tone_levels)
    stimuli = synth.synthesize_sequence(params)
    params = si.stitch_parameters(params, '_input', stimuli)

    # Add model parameters
    params = si.append_parameters(params, ['cf_low', 'cf_high', 'n_cf'], [1000, 1000, 1])

    # Run model
    output = sim.run(params)

    # Assert that output rates grow with frequency
    means = [np.mean(out) for out in output]
    assert np.all(np.diff(means) > 0)
