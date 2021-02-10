import apcmodels.simulation as si
import numpy as np


def test_simulator_construct_batch_size():
    """ Check that construct batch accepts stimulus and parameter sequences of variable sizes and returns a list of the
    appropriate size"""
    sim = si.Simulator()
    batch_zip = sim.construct_batch([np.zeros(5), np.ones(5), 2 * np.ones(5)],
                                    [{'freq': 1000}, {'freq': 2000}, {'freq': 5000}],
                                    [{'CF': 50, 'fs': 10000}, {'CF': 100, 'fs': 10000}, {'CF': 200, 'fs': 10000}], 'zip')
    batch_product = sim.construct_batch([np.zeros(5), np.ones(5), 2 * np.ones(5)],
                                        [{'freq': 1000}, {'freq': 2000}, {'freq': 5000}],
                                        [{'CF': 50, 'fs': 10000}, {'CF': 100, 'fs': 10000}], 'product')
    assert len(batch_zip) == 3 and len(batch_product) == 6


def test_simulator_construct_batch_error_handling():
    """ Check that construct batch correctly throws an error if sizes of inputs don't match """
    sim = si.Simulator()
    try:
        temp = sim.construct_batch([np.zeros(5), np.ones(5), 2 * np.ones(5)], [{'freq': 1000}, {'freq': 2000}],
                                   [{'CF': 50, 'fs': 10000}, {'CF': 100, 'fs': 10000}], 'zip')
        raise Exception('Should have failed!')
    except AssertionError:
        return
    
    
def test_simulator_construct_batch_product():
    """ Check that construct_batch product can successfully apply a single params set to multipel input stimuli """
    # Initialize simulator object
    sim = si.Simulator()
    batch = sim.construct_batch(inputs=[1, 2, 3], input_parameters=[None, None, None], model_parameters=[{'a': 1}])
    assert batch[0]['_input'] == 1 and batch[1]['_input'] == 2 and batch[2]['_input'] == 3


def test_simulator_construct_batch_nested_stimuli():
    """ Check that when construct_batch is provided with a stimulus array that has nested stimuli that
    things are handled correctly """
    sim = si.Simulator()
    batch = sim.construct_batch(inputs=[[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]],
                                input_parameters=[[{'a': 0.1}, {'a': 0.2}, {'a': 0.3}],
                                                  [{'a': 0.1}, {'a': 0.2}, {'a': 0.3}]],
                                model_parameters=[{'fs': 1000}])



def test_simulator_run_simple_batch():
    """ Check that run can handle arbitrary inputs """
    # Create dummy function that returns input as output --- thus, the output of run() should be identical to its input
    def dummy(_input):
        return _input
    # Initialize simulator object
    sim = si.Simulator()
    # Create dummy input
    dummy_input = [{'_input': 1}, {'_input': 2}]
    output = sim.run(batch=dummy_input, runfunc=dummy, parallel=True)
    # Check that all inputs and outputs align
    for this_input, this_output in zip(dummy_input, output):
        assert(this_input['_input'] == this_output)


def test_simulator_run():
    """ Check that run() correctly accepts a list of dicts and returns a corresponding number of responses"""
    sim = si.Simulator()
    results = sim.run([{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}], sim.simulate, parallel=False)
    assert len(results) == 2


def test_simulator_run_parallel():
    """ Check that run() correctly accepts a list of dicts and returns a corresponding number of responses"""
    sim = si.Simulator()
    results = sim.run([{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}], sim.simulate, parallel=True)
    assert len(results) == 2


def test_increment_parameters_input_dict():
    """ Check that increment_parameters accepts a dict as input """
    si.increment_parameters(parameters={'a': 1, 'b': 2}, increments={'a': 0.01})


def test_increment_parameters_input_list():
    """ Check that increment_parameters accepts a list as input """
    si.increment_parameters(parameters=[{'a': 1, 'b': 2}], increments={'a': 0.01})


def test_increment_parameters_input_nested_list():
    """ Check that increment_parameters accepts a nested list as input """
    si.increment_parameters(parameters=[{'a': 1, 'b': 2},
                                        [[{'a': 1, 'b': 2}, {'a': 1, 'b': 2}], [{'a': 2, 'b': 40}]]],
                            increments={'a': 0.01})


def test_wiggle_parameters_input_dict():
    """ Check that wiggle_parameters accepts a dict as input """
    si.wiggle_parameters(parameters={'a': 1, 'b': 2}, parameter_to_wiggle='a', values=[1, 2, 3, 4])


def test_wiggle_parameters_input_list():
    """ Check that wiggle_parameters accepts a list as input """
    si.wiggle_parameters(parameters=[{'a': 1, 'b': 2}], parameter_to_wiggle='a', values=[1, 2, 3, 4])


def test_wiggle_parameters_input_nested_list():
    """ Check that wiggle_parameters accepts a nested list as input """
    si.wiggle_parameters(parameters=[{'a': 1, 'b': 2},
                                     [[{'a': 1, 'b': 2}, {'a': 1, 'b': 2}], [{'a': 2, 'b': 40}]]],
                         parameter_to_wiggle='a', values=[1, 2, 3, 4])


def test_wiggle_parameters_repeated():
    """ Check that increment_parameters accepts a nested list as input """
    si.wiggle_parameters(parameters=si.wiggle_parameters(parameters={'a': 1, 'b': 2},
                                                         parameter_to_wiggle='a',
                                                         values=[1, 2, 3, 4]),
                         parameter_to_wiggle='b',
                         values=[10, 20])