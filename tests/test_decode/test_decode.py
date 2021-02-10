from apcmodels.decode import *
import apcmodels.synthesis as sy
import apcmodels.simulation as si
import apcmodels.anf as anf


def test_run_rates_util():
    """ Test to make sure that run_rates_util will correctly accept either a single input or a list of inputs and return
    the right output """
    def dummy_ratefunc(_input):
        return _input

    # Test to make sure that if you just provide it with single input it handles that okay
    output1 = run_rates_util(dummy_ratefunc, _input=1)
    assert output1 == 1

    # Also test that it handles a list and returns the input
    output2 = run_rates_util(dummy_ratefunc, _input=[1, 2, 3, 4])
    for _in, _out in zip([1, 2, 3, 4], output2):
        assert _in == _out


def test_ideal_observer_single_input():
    """ Test that if we provide a single stimulus to a ratefunc wrapped in decode_ideal_observer that some sort of
     error is raised to indicate that an ideal observer can't be calculated based on a single simulation! """
    # Initialize simulator object
    sim = anf.AuditoryNerveHeinz2001Numba()

    # Define stimulus parameters
    fs = int(200e3)
    def tone_level(): return np.random.uniform(25, 35, 1)
    tone_dur = 0.1
    tone_ramp_dur = 0.01
    tone_freq = 1000

    # Synthesize stimuli
    synth = sy.PureTone()
    params = {'level': tone_level, 'dur': tone_dur, 'dur_ramp': tone_ramp_dur, 'freq': tone_freq, 'fs': fs}
    stimuli = synth.synthesize_sequence([params])

    # Define model
    params_model = [{'cf_low': 1000, 'cf_high': 1000, 'n_cf': 1}]
    batch = sim.construct_batch(inputs=stimuli, input_parameters=[params],
                                model_parameters=params_model, mode='zip')
    batch = si.append_parameters(batch, 'fs', int(200e3))
    batch = si.append_parameters(batch, 'n_fiber_per_chan', 5)
    batch = si.append_parameters(batch, 'delta_theta', [0.001])
    batch = si.append_parameters(batch, 'API', np.zeros(1))

    # Run model
    try:
        sim.run(batch=[batch], parallel=True, runfunc=decode_ideal_observer(sim.simulate))
        raise Exception('This should have failed!')
    except ValueError:
        return


def test_ideal_observer_single_input():
    """ Test that if we provide a single stimulus to a ratefunc wrapped in decode_ideal_observer that some sort of
     error is raised to indicate that an ideal observer can't be calculated based on a single simulation! """
    # Initialize simulator object
    sim = anf.AuditoryNerveHeinz2001Numba()

    # Define stimulus parameters
    fs = int(200e3)
    def tone_level(): return np.random.uniform(25, 35, 1)
    tone_dur = 0.1
    tone_ramp_dur = 0.01
    tone_freq = 1000

    # Synthesize stimuli
    synth = sy.PureTone()
    params = {'level': tone_level, 'dur': tone_dur, 'dur_ramp': tone_ramp_dur, 'freq': tone_freq, 'fs': fs}
    stimuli = synth.synthesize_sequence([params])

    # Define model
    params_model = [{'cf_low': 1000, 'cf_high': 1000, 'n_cf': 1}]

    # Run model
    try:
        sim.run_batch(inputs=stimuli, input_parameters=[params], model_parameters=params_model,
                      runfunc=decode_ideal_observer(sim.simulate), mode='zip',
                      parameters_to_append={'fs': int(200e3),
                                           'n_fiber_per_chan': 5,
                                           'delta_theta': [0.001],
                                           'API': np.zeros(1)})
        raise Exception('This should have failed!')
    except ValueError:
        return