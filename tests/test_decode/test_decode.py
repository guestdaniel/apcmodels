from apcmodels.decode import *
import apcmodels.anf as anf

def test_ideal_observer():
    """ Make sure that we can wrap a standard ratefunc in decode ideal observer and get decoded thresholds out """
    def dummy_ratefunc(_input, **kwargs):
        return _input

    kwargs = {'_input': [0, 1, 2, 3, 4], 'other_param': 2}
    output = decode_ideal_observer(dummy_ratefunc)(**kwargs)
    assert len(output) == 2

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