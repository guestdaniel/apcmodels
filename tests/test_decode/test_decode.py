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
    assert output1 == [1]

    # Also test that it handles a list and returns the input
    output2 = run_rates_util(dummy_ratefunc, _input=[1, 2, 3, 4])
    for _in, _out in zip([1, 2, 3, 4], output2):
        assert _in == _out

def test_ideal_observer_real_simulation():
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
