import apcmodels.simulation as sy
import numpy as np
import apcmodels.synthesis


def test_simulator_construct_batch_size():
    """ Check that construct batch accepts stimulus and parameter sequences of variable sizes and returns a list of the
    appropriate size"""
    sim = sy.Simulator()
    batch_zip = sim.construct_batch([np.zeros(5), np.ones(5), 2 * np.ones(5)],
                                    [{'freq': 1000}, {'freq': 2000}, {'freq': 5000}],
                                    [{'CF': 50, 'fs': 10000}, {'CF': 100, 'fs': 10000}, {'CF': 200, 'fs': 10000}], 'zip')
    batch_product = sim.construct_batch([np.zeros(5), np.ones(5), 2 * np.ones(5)],
                                        [{'freq': 1000}, {'freq': 2000}, {'freq': 5000}],
                                        [{'CF': 50, 'fs': 10000}, {'CF': 100, 'fs': 10000}], 'product')
    assert len(batch_zip) == 3 and len(batch_product) == 6


def test_simulator_construct_batch_error_handling():
    """ Check that construct batch correctly throws an error if sizes of inputs don't match """
    sim = sy.Simulator()
    try:
        temp = sim.construct_batch([np.zeros(5), np.ones(5), 2 * np.ones(5)], [{'freq': 1000}, {'freq': 2000}],
                                   [{'CF': 50, 'fs': 10000}, {'CF': 100, 'fs': 10000}], 'zip')
        raise Exception('Should have failed!')
    except AssertionError:
        return
    
    
def test_simulator_construct_batch_product():
    """ Check that construct_batch product can successfully apply a single params set to multipel input stimuli """
    # Initialize simulator object
    sim = sy.Simulator()
    batch = sim.construct_batch(inputs=[1, 2, 3], input_parameters=[None, None, None], model_parameters=[{'a': 1}])
    assert batch[0]['input'] == 1 and batch[1]['input'] == 2 and batch[2]['input'] == 3


def test_simulator_run_simple_batch():
    """ Check that run can handle arbitrary inputs """
    # Create dummy function that returns input as output --- thus, the output of run() should be identical to its input
    def dummy(input):
        return input
    # Initialize simulator object
    sim = sy.Simulator()
    # Create dummy input
    dummy_input = ['yo', 'ye', 'ya']
    output = sim.run(batch=dummy_input, runfunc=dummy, parallel=True)
    # Check that all inputs and outputs align
    for this_input, this_output in zip(dummy_input, output):
        assert(this_input == this_output)


def test_simulator_run():
    """ Check that run() correctly accepts a list of dicts and returns a corresponding number of responses"""
    sim = sy.Simulator()
    results = sim.run([{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}], sim.simulate, parallel=False)
    assert len(results) == 2


def test_simulator_run_parallel():
    """ Check that run() correctly accepts a list of dicts and returns a corresponding number of responses"""
    sim = sy.Simulator()
    results = sim.run([{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}], sim.simulate, parallel=True)
    assert len(results) == 2
