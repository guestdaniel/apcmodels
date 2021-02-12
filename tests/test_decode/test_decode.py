from apcmodels.decode import *
import apcmodels.synthesis as sy
import apcmodels.simulation as si
import apcmodels.anf as anf


def test_run_rates_util():
    """ Test to make sure that run_rates_util will correctly accept either a single input or a list of inputs and return
    the right output """
    def dummy_ratefunc(params):
        return params['_input']

    # Test to make sure that if you just provide it with single input it handles that okay
    output1 = run_rates_util(dummy_ratefunc, {'_input': 1})
    assert output1 == 1

    # Also test that it handles a list and returns the input
    output2 = run_rates_util(dummy_ratefunc, [{'_input': 1}, {'_input': 2}, {'_input': 3}])
    for _in, _out in zip([1, 2, 3], output2):
        assert _in == _out


def test_ideal_observer_single_input():
    """ Test that if we provide a single stimulus to a ratefunc wrapped in decode_ideal_observer that some sort of
     error is raised to indicate that an ideal observer can't be calculated based on a single simulation! """
    # Initialize simulator object
    sim = anf.AuditoryNerveHeinz2001Numba()

    # Define stimulus parameters
    fs = int(200e3)
    tone_level = 30
    tone_dur = 0.1
    tone_ramp_dur = 0.01
    tone_freq = 1000

    # Synthesize stimuli
    synth = sy.PureTone()
    params = {'level': tone_level, 'dur': tone_dur, 'dur_ramp': tone_ramp_dur, 'freq': tone_freq, 'fs': fs}
    stimuli = synth.synthesize_sequence([params])

    # Add stimuli and model params
    params = si.append_parameters(params, ['_input', 'cf_low', 'cf_high', 'n_cf'], [stimuli[0], 1000, 1000, 1])
    params = si.append_parameters(params, ['n_fiber_per_chan', 'fs', 'delta_theta', 'API'],
                                  [5, int(200e3), [0.001], np.zeros(1)])

    # Run model
    try:
        sim.run([params], runfunc=decode_ideal_observer(sim.simulate))
        raise Exception('This should have failed!')
    except ValueError:
        return


def test_find_parameter():
    """ Test to make sure that find_parameter accepts a nested list of params and returns the appropriate value """
    params = [[{'a': 5, 'b': 3}], [{'c': 2}]]
    assert find_parameter(params, 'c') == 2


def test_find_parameter_error():
    """ Test to make sure that find_parameter returns a raises an error when we can't find a given parameter """
    params = [[{'a': 5, 'b': 3}], [{'c': 2}]]
    try:
        find_parameter(params, 'd')
        raise Exception('This should have failed!')
    except LookupError:
        return