from apcmodels.decode import *

def test_run_rates_util():
    def dummy_ratefunc(_input):
        return _input

    # Test to make sure that if you just provide it with single input it handles that okay
    output1 = run_rates_util(dummy_ratefunc, _input=1)
    assert output1 == [1]

    # Also test that it handles a list and returns the input
    output2 = run_rates_util(dummy_ratefunc, _input=[1, 2, 3, 4])
    for _in, _out in zip([1, 2, 3, 4], output2):
        assert _in == _out
