from apcmodels.decode import *
import apcmodels.synthesis as sy
import apcmodels.simulation as si
import apcmodels.anf as anf


def test_ideal_observer_FDL_vs_frequency():
    """ Test that ideal observer analysis on simple pure tone FDLs shows increasing FDLs with increasing frequency """
    # Initialize simulator object
    sim = anf.AuditoryNerveHeinz2001()

    # Define stimulus parameters
    fs = int(200e3)
    tone_level = 30
    tone_dur = 0.1
    tone_ramp_dur = 0.01
    tone_freqs = [1000, 2000, 4000, 8000]

    # Encode stimulus information
    params = {'level': tone_level, 'dur': tone_dur, 'dur_ramp': tone_ramp_dur, 'fs': fs}
    params = si.wiggle_parameters(params, 'freq', tone_freqs)

    # Encode model information
    params = si.stitch_parameters(params, 'cf_low', [1000, 2000, 4000, 8000])
    params = si.stitch_parameters(params, 'cf_high', [1000, 2000, 4000, 8000])
    params = si.append_parameters(params, ['n_cf', 'fs', 'n_fiber_per_chan', 'delta_theta', 'API'],
                                  [1, int(200e3), 5, [0.001], np.zeros((1))])

    # Flatten and increment frequency
    params = si.flatten_parameters(params)
    params = si.increment_parameters(params, {'freq': 0.001})
    synth = sy.PureTone()
    stimuli = synth.synthesize_sequence(params)
    params = si.stitch_parameters(params, '_input', stimuli)

    # Run model
    output = sim.run(params,
                     parallel=True,
                     runfunc=decode_ideal_observer(sim.simulate))

    # Extract AI thresholds
    output = [out[0] for out in output]  # AI thresholds are always the first element of each tuple in output

    # Check to make sure that thresholds grow with frequency
    assert np.all(np.diff(output) > 0)


def test_ideal_observer_real_simulation_with_level_roving():
    """ Test that ideal observer analysis on simple pure tone FDLs shows increasing FDLs with increasing frequency in
    the context of a mild level rove on the pure tone """
    # Initialize simulator object
    sim = anf.AuditoryNerveHeinz2001()

    # Define stimulus parameters
    fs = int(200e3)
    def tone_level(): return np.random.uniform(25, 35, 1)
    tone_dur = 0.1
    tone_ramp_dur = 0.01
    tone_freqs = [1000, 2000, 4000, 8000]

    # Encode stimulus parameters
    params = {'level': tone_level, 'dur': tone_dur, 'dur_ramp': tone_ramp_dur, 'fs': fs}
    params = si.wiggle_parameters(params, 'freq', tone_freqs)

    # Encode model parameters
    params = si.stitch_parameters(params, 'cf_low', [1000, 2000, 4000, 8000])
    params = si.stitch_parameters(params, 'cf_high', [1000, 2000, 4000, 8000])
    params = si.append_parameters(params, ['fs', 'n_cf', 'n_fiber_per_chan', 'delta_theta', 'API'],
                                  [int(200e3), 1, 5, [0.001, 0.001], np.array([[0, 0], [0, 1/6**2]])])

    # Encode repeats and increments
    params = si.repeat_parameters(params, 10)
    params = si.increment_parameters(params, {'freq': 0.001, 'level': 0.001})

    # Synthesize stimuli and encode in params
    synth = sy.PureTone()
    stimuli = synth.synthesize_sequence(params)
    params = si.stitch_parameters(params, '_input', stimuli)

    # Run model
    output = sim.run(params,
                     parallel=True,
                     runfunc=decode_ideal_observer(sim.simulate))

    # Extract AI thresholds
    output = [out[0] for out in output]  # AI thresholds are always the first element of each tuple in output

    # Check to make sure that thresholds grow with frequency
    assert np.all(np.diff(output) > 0)