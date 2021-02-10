from apcmodels.decode import *
import apcmodels.synthesis as sy
import apcmodels.simulation as si
import apcmodels.anf as anf


def test_ideal_observer_FDL_vs_frequency():
    """ Test that ideal observer analysis on simple pure tone FDLs shows increasing FDLs with increasing frequency """
    # Initialize simulator object
    sim = anf.AuditoryNerveHeinz2001Numba()

    # Define stimulus parameters
    fs = int(200e3)
    tone_level = 30
    tone_dur = 0.1
    tone_ramp_dur = 0.01
    tone_freqs = [1000, 2000, 4000, 8000]

    # Synthesize stimuli
    synth = sy.PureTone()
    params = {'level': tone_level, 'dur': tone_dur, 'dur_ramp': tone_ramp_dur, 'fs': fs}
    params = si.wiggle_parameters(params, 'freq', tone_freqs)
    params = si.increment_parameters(params, {'freq': 0.001})
    stimuli = synth.synthesize_sequence(params)

    # Define model
    params_model = [{'cf_low': 1000, 'cf_high': 1000, 'n_cf': 1},
                    {'cf_low': 2000, 'cf_high': 2000, 'n_cf': 1},
                    {'cf_low': 4000, 'cf_high': 4000, 'n_cf': 1},
                    {'cf_low': 8000, 'cf_high': 8000, 'n_cf': 1}]
    batch = sim.construct_batch(inputs=stimuli, input_parameters=params,
                                model_parameters=params_model, mode='zip')
    batch = si.append_parameters(batch, 'fs', int(200e3))
    batch = si.append_parameters(batch, 'n_fiber_per_chan', 5)
    batch = si.append_parameters(batch, 'delta_theta', [0.001])
    batch = si.append_parameters(batch, 'API', np.zeros(1))

    # Run model
    output = sim.run(batch=batch,
                     parallel=True,
                     runfunc=decode_ideal_observer(sim.simulate))

    # Check to make sure that thresholds grow with frequency
    assert np.all(np.diff(output) > 0)


def test_ideal_observer_real_simulation_with_level_roving():
    """ Test that ideal observer analysis on simple pure tone FDLs shows increasing FDLs with increasing frequency in
    the context of a mild level rove on the pure tone """
    # Initialize simulator object
    sim = anf.AuditoryNerveHeinz2001Numba()

    # Define stimulus parameters
    fs = int(200e3)
    def tone_level(): return np.random.uniform(25, 35, 1)
    tone_dur = 0.1
    tone_ramp_dur = 0.01
    tone_freqs = [1000, 2000, 4000, 8000]

    # Synthesize stimuli
    synth = sy.PureTone()
    params = {'level': tone_level, 'dur': tone_dur, 'dur_ramp': tone_ramp_dur, 'fs': fs}
    params = si.wiggle_parameters(params, 'freq', tone_freqs)
    params = si.repeat(params, 10)
    params = si.increment_parameters(params, {'freq': 0.001, 'level': 0.001})
    stimuli = synth.synthesize_sequence(params)

    # Define model
    params_model = [{'cf_low': 1000, 'cf_high': 1000, 'n_cf': 4},
                    {'cf_low': 2000, 'cf_high': 2000, 'n_cf': 4},
                    {'cf_low': 4000, 'cf_high': 4000, 'n_cf': 4},
                    {'cf_low': 8000, 'cf_high': 8000, 'n_cf': 4}]
    batch = sim.construct_batch(inputs=stimuli, input_parameters=params,
                                model_parameters=params_model, mode='zip')
    batch = si.append_parameters(batch, 'fs', int(200e3))
    batch = si.append_parameters(batch, 'n_fiber_per_chan', 5)
    batch = si.append_parameters(batch, 'delta_theta', [0.001, 0.001])
    batch = si.append_parameters(batch, 'API', np.array([[0, 0], [0, 1/6**2]]))

    # Run model
    output = sim.run(batch=batch,
                     parallel=True,
                     runfunc=decode_ideal_observer(sim.simulate))

    # Check to make sure that thresholds grow with frequency
    assert np.all(np.diff(output) > 0)