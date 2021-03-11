import apcmodels.simulation as si
import numpy as np


def test_parameters_init():
    """ Test that Parameters.__init__() accepts kwargs that are turned into entries in its params attribute """
    params = si.Parameters(hello='world', foo='bar')
    assert params[0]['hello'] == 'world' and params[0]['foo'] == 'bar'


def test_parameters_append():
    """ Test that Parameters.append() accepts a single keyword argument and correctly appends it. """
    params = si.Parameters(hello='world')
    params.append('foo', 'bar')
    assert params[0]['hello'] == 'world' and params[0]['foo'] == 'bar'


def test_parameters_append_2d():
    """ Test that Parameters.append() accepts a single keyword argument and correctly appends it with the Parameters 
    object is 2D """
    params = si.Parameters(hello='world')
    params.wiggle('hello', ['world', 'mundo'])
    params.wiggle('goodbye', ['world', 'mundo'])
    params.append('foo', 'bar')
    assert params[0, 0]['hello'] == 'world' and params[1, 1]['goodbye'] == 'mundo'


def test_parameters_add_inputs():
    """ Test that Parameters.add_inputs() correctly accepts an array object and adds it with the key '_input' """
    params = si.Parameters(hello='world')
    params.add_inputs(np.array(['bar']))
    assert params[0]['hello'] == 'world' and params[0]['_input'] == 'bar'


def test_parameters_combine():
    """ Test that Parameters.add_inputs() correctly accepts an array object and adds it with the key
    '_input' """
    params = si.Parameters(hello='world')
    params.combine(si.Parameters(foo='bar'))
    assert params[0]['hello'] == 'world' and params[0]['foo'] == 'bar'


def test_parameters_evaluate():
    """ Test that Parameters.evaluate() correctly evaluates callable element in Parameters """
    def tempfunc():
        return 'world'
    params = si.Parameters(hello=tempfunc)
    params.evaluate()
    assert params[0]['hello'] == 'world'


def test_parameters_flatten():
    """ Test that Parameters.flatten() correctly flattens a 3D parameters object """
    params = si.Parameters()
    params.wiggle('a', [1, 2, 3])
    params.wiggle('b', [1, 2, 3])
    params.wiggle('c', [1, 2, 3])
    params.flatten()
    assert params.shape == (3*3*3, )


def test_parameters_flatten_with_nested_structure():
    """ Test that Parameters.flatten() correctly flattens a 3D parameters object with at least one nested element """
    params = si.Parameters()
    params.wiggle('a', [1, 2, 3])
    params.wiggle('b', [1, 2, 3])
    params.wiggle('c', [1, 2, 3])
    params[0, 0, 0] = [params[0, 0, 0], dict()]
    params.flatten()
    assert params.shape == (3*3*3, ) and len(params[0]) == 2


def test_parameters_increment():
    """ Test that Parameters.increment() correctly increments a single parameter """
    params = si.Parameters(hello=1)
    params.increment({'hello': 0.01})
    assert params[0][0]['hello'] == 1 and params[0][1]['hello'] == 1.01


def test_parameters_repeat():
    """ Test that Parameters.repeat() correctly repeats an encoded simulation """
    params = si.Parameters(hello='world')
    params.repeat(2)
    assert params[0][0]['hello'] == 'world' and params[0][1]['hello'] == 'world'


def test_parameters_stitch():
    """ Test that Parameters.stitch() stitches an input array into Parameters """
    params = si.Parameters(hello='world')
    params.stitch('foo', np.array(['bar']))
    assert params[0]['hello'] == 'world' and params[0]['foo'] == 'bar'


def test_parameters_stitch_nested():
    """ Test that Parameters.stitch() stitches an input array into Parameters even for a Parameters object with nested
    elements (i.e., a list of dicts instead of a dict) """
    params = si.Parameters(hello='world')
    params.repeat(5)
    x = np.empty((1,), dtype=object)
    x[0] = [1, 2, 3, 4, 5]
    params.add_inputs(x)


def test_parameters_wiggle():
    """ Test that Parameters.wiggle() wiggles a single parameter correctly """
    params = si.Parameters(hello='world')
    params.wiggle('foo', [1, 2, 3])
    assert params[0]['foo'] == 1 and params[2]['foo'] == 3


def test_parameters_wiggle_parallel():
    """ Test that Parameters.wiggle_parallel() wiggles two parameters correctly """
    params = si.Parameters(hello='world')
    params.wiggle_parallel(['foo', 'bloop'], [[1, 2, 3], [4, 5, 6]])
    assert params[0]['foo'] == 1 and params[2]['foo'] == 3 and params[0]['bloop'] == 4 and params[2]['bloop'] == 6


def test_parameters_array_like_behavior():
    """ Test that Parameters exhibits the correct array-like behavior """
    params = si.Parameters()
    assert params.shape == (1, )                                  # check that params has shape
    assert params.size == 1                                       # check that params has size
    assert params.__repr__() == 'array([{}], dtype=object)'       # check that params has __str__
    assert params.__str__() == '[{}]'                             # check that params has __str__
    params[0] = 1                                                 # check that params has __setitem__
    assert params[0] == 1


def test_simulator_run_simple_batch():
    """ Test that run can handle arbitrary inputs """
    # Create dummy function that returns input as output --- thus, the output of run() should be identical to its input
    def dummy(params):
        return params['_input']
    # Initialize simulator object
    sim = si.Simulator()
    # Create dummy input
    dummy_input = [{'_input': 1}, {'_input': 2}]
    output = sim.run(params=dummy_input, runfunc=dummy, parallel=True)
    # Test that all inputs and outputs align
    for this_input, this_output in zip(dummy_input, output):
        assert(this_input['_input'] == this_output)


def test_simulator_run():
    """ Test that run() correctly accepts a list of dicts and returns a corresponding number of responses"""
    sim = si.Simulator()
    results = sim.run([{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}], sim.simulate, parallel=False)
    assert len(results) == 2


def test_simulator_run_parallel():
    """ Test that run() correctly accepts a list of dicts and returns a corresponding number of responses """
    sim = si.Simulator()
    results = sim.run([{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}], sim.simulate, parallel=True)
    assert len(results) == 2


def test_simulator_run_array():
    """ Test that run() correctly accepts an array of dicts and returns a corresponding number of responses """
    sim = si.Simulator()
    results = sim.run(np.array([{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}]), sim.simulate, parallel=False)
    assert results.shape == (2,)


def test_simulator_run_array_2d():
    """ Test that run() correctly accepts a 2d array of dicts and returns a corresponding 2d array of responses """
    sim = si.Simulator()
    results = sim.run(np.array([[{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}],
                                [{'foo': 1, 'bar': 20}, {'foo': 3, 'bar': 40}]]), parallel=False)
    assert results.shape == (2, 2)


def test_simulator_run_array_parallel():
    """ Test that run() correctly accepts an array of dicts and returns a corresponding number of responses while
    using multiprocessing """
    sim = si.Simulator()
    results = sim.run(np.array([{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}]), sim.simulate, parallel=True)
    assert results.shape == (2,)


def test_simulator_run_array_2d_parallel():
    """ Test that run() correctly accepts a 2d array of dicts and returns a corresponding 2d array of responses while
    using multiprocessing """
    sim = si.Simulator()
    results = sim.run(np.array([[{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}],
                                [{'foo': 1, 'bar': 20}, {'foo': 3, 'bar': 40}]]), parallel=False)
    assert results.shape == (2, 2)


def test_simulator_wrong_inputs_parallel():
    """ Test that run() raises an error if we pass wrong types into params """
    sim = si.Simulator()
    try:
        results = sim.run(123, sim.simulate, parallel=True)
        raise Exception('This should have failed!')
    except TypeError:
        return


def test_simulator_wrong_inputs():
    """ Test that run() raises an error if we pass wrong types into params """
    sim = si.Simulator()
    try:
        results = sim.run(123, sim.simulate, parallel=False)
        raise Exception('This should have failed!')
    except TypeError:
        return


def test_append_parameters_single_element_input():
    """ Test that if we provide append_parameters with an input with just one element that it still returns a list """
    assert type(si.append_parameters([{'hello': 'world'}], 'foo', 'bar')) == list


def test_append_parameters_single_item_to_append():
    """ Test that if we provide append_parameters with an reasonable input (a list of multiple dicts) that it correctly
     appends a parameter to each dict """
    params = [{'x': 1}, {'x': 2}, {'x': 3}]
    params = si.append_parameters(params, 'y', 2)
    assert params[0]['y'] == 2 and params[1]['y'] == 2


def test_append_parameters_multiple_items_to_append():
    """ Test that if we provide append_parameters with an reasonable input (a list of multiple dicts) that it correctly
     appends multiple parameters to each dict """
    params = [{'x': 1}, {'x': 2}, {'x': 3}]
    params = si.append_parameters(params, ['y', 'z'], [2, 3])
    assert params[0]['y'] == 2 and params[1]['y'] == 2 and params[0]['z'] == 3 and params[1]['z'] == 3


def test_append_parameters_bad_inputs1():
    """ Test that if we provide append_parameters with an reasonable input (a list of multiple dicts) but bad inputs
     ( one list and one not list) that it raises the right errors """
    try:
        si.append_parameters({'x': 1}, ['y'], 2)
        raise Exception('This should have failed!')
    except ValueError:
        return


def test_append_parameters_bad_inputs2():
    """ Test that if we provide append_parameters with an reasonable input (a list of multiple dicts) but bad inputs
     (two lists of different lengths) that it raises the right errors """
    try:
        si.append_parameters({'x': 1}, ['y'], [1, 2, 3])
        raise Exception('This should have failed!')
    except ValueError:
        pass


def test_append_parameters_bad_inputs3():
    """ Test that if we provide append_parameters with an input of wrong type that we get a warning """
    try:
        si.append_parameters('x', ['y'], [1, 2, 3])
        raise Exception('This should have failed!')
    except TypeError:
        pass


def test_append_parameters_single_item_to_append_array():
    """ Test that if we provide append_parameters with an reasonable input (an array of multiple dicts) that it
    correctly appends a parameter to each dict """
    params = np.array([{'x': 1}, {'x': 2}, {'x': 3}])
    params = si.append_parameters(params, 'y', 2)
    assert params[0]['y'] == 2 and params[1]['y'] == 2


def test_append_parameters_multiple_items_to_append_array():
    """ Test that if we provide append_parameters with an reasonable input (an array of multiple dicts) that it
     correctly appends multiple parameters to each dict """
    params = np.array([{'x': 1}, {'x': 2}, {'x': 3}])
    params = si.append_parameters(params, ['y', 'z'], [2, 3])
    assert params[0]['y'] == 2 and params[1]['y'] == 2 and params[0]['z'] == 3 and params[1]['z'] == 3


def test_combine_parameters_list_len():
    """ Test that if we provide combine_parameters with inputs of the wrong lengths that we get an error raised """
    try:
        params = si.combine_parameters([{'a': 1}], [{'b': 2}, {'b': 3}])
        raise Exception('This should have failed!')
    except ValueError:
        return


def test_combine_parameters_array_shape():
    """ Test that if we provide combine_parameters with inputs of the wrong lengths that we get an error raised """
    try:
        params = si.combine_parameters(np.array([{'a': 1}]), np.array([{'b': 2}, {'b': 3}]))
        raise Exception('This should have failed!')
    except ValueError:
        return


def test_combine_parameters_list():
    """ Test that if we provide combine_parameters with an reasonable input (two lists of dicts) that it correctly
    correctly combines them """
    params = si.combine_parameters([{'a': 1}], [{'b': 2}])
    assert params[0]['a'] == 1 and params[0]['b'] == 2


def test_combine_parameters_array():
    """ Test that if we provide combine_parameters with an reasonable input (two lists of dicts) that it correctly
    correctly combines them """
    params = si.combine_parameters(np.array([{'a': 1}]), np.array([{'b': 2}]))
    assert params[0]['a'] == 1 and params[0]['b'] == 2


def test_evaluate_parameters_bad_input():
    """ Test that if we provide evaluate_parameters wtih an input of the wrong type that we get can error """
    try:
        si.evaluate_parameters(123)
        raise Exception('This should have failed!')
    except TypeError:
        return


def test_increment_parameters_input_dict():
    """ Test that increment_parameters accepts a dict as input """
    results = si.increment_parameters(parameters={'a': 1, 'b': 2}, increments={'a': 0.01})
    assert results[0]['a'] == 1 and results[1]['a'] == 1.01


def test_increment_parameters_input_list():
    """ Test that increment_parameters accepts a list as input """
    results = si.increment_parameters(parameters=[{'a': 1, 'b': 2}], increments={'a': 0.01})
    assert results[0][0]['a'] == 1 and results[0][1]['a'] == 1.01


def test_increment_parameters_input_nested_list():
    """ Test that increment_parameters accepts a nested list as input """
    results = si.increment_parameters(parameters=[{'a': 1, 'b': 2},
                                                  [[{'a': 1, 'b': 2}, {'a': 1, 'b': 2}], [{'a': 2, 'b': 40}]]],
                                      increments={'a': 0.01})
    assert results[1][0][0][0]['a'] == 1 and results[1][0][0][1]['a'] == 1.01


def test_increment_parameters_array():
    """ Test that increment_parameters accepts an array as input """
    results = si.increment_parameters(parameters=np.array([{'a': 0.01}]),
                                      increments={'a': 0.01})
    assert results[0][0]['a'] == 0.01 and results[0][1]['a'] == 0.02


def test_increment_parameters_input_dict_with_repeated_increment_value():
    """ Test that increment_parameters correctly handles parameter names appended with (#)_ """
    results = si.increment_parameters(parameters={'a': 1, 'b': 2}, increments={'(1)_a': 0.01, '(2)_a': 0.02})
    assert results[0]['a'] == 1 and results[1]['a'] == 1.01 and results[2]['a'] == 1.02


def test_increment_parameters_input_dict_with_repeated_increment_value_callable_value():
    """ Test that increment_parameters correctly handles parameter names appended with (#)_ when the associated
    parameter value is callable """
    results = si.increment_parameters(parameters={'a': lambda: 1, 'b': 2}, increments={'(1)_a': 0.01, '(2)_a': 0.02})
    assert results[0]['a'] == 1 and results[1]['a'] == 1.01 and results[2]['a'] == 1.02


def test_repeat_parameters_bad_input():
    """ Test that repeat_parameters raises an error if we pass an input of the wrong type """
    try:
        results = si.repeat_parameters(1, 5)
        raise Exception('This should have failed!')
    except TypeError:
        return


def test_repeat_parameters():
    """ Test that repeat_parameters accepts a dict as input """
    results = si.repeat_parameters({'a': 1}, 5)
    assert results[0]['a'] == 1 and results[4]['a'] == 1


def test_repeat_parameters_list():
    """ Test that repeat_parameters accepts a list as input """
    results = si.repeat_parameters([{'a': 1}], 5)
    assert results[0][0]['a'] == 1 and results[0][4]['a'] == 1


def test_repeat_parameters_array():
    """ Test that repeat_parameters accepts an array as input """
    results = si.repeat_parameters(np.array([{'a': 1}]), 5)
    assert results[0][0]['a'] == 1 and results[0][4]['a'] == 1


def test_stitch_parameters_bad_inputs():
    """ Test that stitch_parameters throws an error if we pass a weird type """
    try:
        results = si.stitch_parameters(1, 'c', [1, 2])
        raise Exception('This should have failed')
    except TypeError:
        return


def test_stitch_parameters_list():
    """ Test that stitch_parameters accepts a list as input and correctly stiches a single parameter """
    results = si.stitch_parameters([{'a': 1, 'b': 2}, {'a': 1, 'b': 2}], 'c', [1, 2])
    assert results[0]['c'] == 1 and results[1]['c'] == 2


def test_stitch_parameters_array():
    """ Test that stitch_parameters accepts an array as input and correctly stiches a single parameter """
    results = si.stitch_parameters(np.array([{'a': 1, 'b': 2}, {'a': 1, 'b': 2}]), 'c', np.array([1, 2]))
    assert results[0]['c'] == 1 and results[1]['c'] == 2


def test_stitch_parameters_list_error_gen():
    """ Test that stitch_parameters accepts a list as input and correctly raises an error if the list lens don't
     match """
    try:
        si.stitch_parameters([{'a': 1, 'b': 2}, {'a': 1, 'b': 2}], 'c', [1])
        raise Exception('this should have failed')
    except ValueError:
        return


def test_stitch_parameters_array_error_gen_size():
    """ Test that stitch_parameters accepts a array as input and correctly raises an error if the array sizes don't
    match """
    try:
        si.stitch_parameters(np.array([{'a': 1, 'b': 2}, {'a': 1, 'b': 2}]), 'c', np.array([1]))
        raise Exception('this should have failed')
    except ValueError:
        return


def test_stitch_parameters_array_error_gen():
    """ Test that stitch_parameters accepts an array as input and correctly stitches new parameters """
    parameters = np.array([[{'a': 1}, {'a': 2}],
                           [{'a': 3}, {'a': 4}]])
    newvals = np.array([[10, 10],
                        [10, 10]])
    output = si.stitch_parameters(parameters, 'b', newvals)
    assert output[0, 0]['b'] == 10 and output[1, 1]['b'] == 10


def test_stitch_parameters_array_2d():
    """ Test that stitch_parameters accepts a 2D parameter as input and correctly stiches a single parameter """
    results = si.stitch_parameters(np.array([{'a': 1, 'b': 2}, {'a': 1, 'b': 2}]), 'c', np.array([1, 2]))
    assert results[0]['c'] == 1 and results[1]['c'] == 2


def test_wiggle_parameters_bad_input():
    """ Test that wiggle_parameters throws an error if we pass a weird type """
    try:
        output = si.wiggle_parameters(1, parameter_to_wiggle='a', values=[1, 2, 3, 4])
        raise Exception('This should have failed!')
    except TypeError:
        return


def test_wiggle_parameters_input_dict():
    """ Test that wiggle_parameters accepts a dict as input and returns sensible output """
    output = si.wiggle_parameters(parameters={'a': 1, 'b': 2}, parameter_to_wiggle='a', values=[1, 2, 3, 4])
    assert output[0]['a'] == 1 and output[3]['a'] == 4


def test_wiggle_parameters_input_list():
    """ Test that wiggle_parameters accepts a list as input and returns sensible output """
    output = si.wiggle_parameters(parameters=[{'a': 1, 'b': 2}], parameter_to_wiggle='a', values=[1, 2, 3, 4])
    assert output[0][0]['a'] == 1 and output[0][3]['a'] == 4


def test_wiggle_parameters_input_array():
    """ Test that wiggle_parameters accepts an array as input and returns sensible output """
    output = si.wiggle_parameters(parameters=np.array([{'a': 1, 'b': 2}]), parameter_to_wiggle='a', values=[1, 2, 3, 4])
    assert output[0]['a'] == 1 and output[3]['a'] == 4


def test_wiggle_parameters_input_array_to_3d_array():
    """ Test that wiggle_parameters accepts an array as input and returns an array of the correct shape once we
     wiggle a few times """
    output = si.wiggle_parameters(parameters=np.array([{'a': 1, 'b': 2}]), parameter_to_wiggle='a', values=[1, 2, 3, 4])
    output = si.wiggle_parameters(output, parameter_to_wiggle='b', values=[1, 2, 3, 4, 5])
    output = si.wiggle_parameters(output, parameter_to_wiggle='c', values=[1, 2, 3, 4, 5, 6])
    assert output.shape == (4, 5, 6)


def test_wiggle_parameters_input_nested_list():
    """ Test that wiggle_parameters accepts a nested list as input """
    si.wiggle_parameters(parameters=[{'a': 1, 'b': 2},
                                     [[{'a': 1, 'b': 2}, {'a': 1, 'b': 2}], [{'a': 2, 'b': 40}]]],
                         parameter_to_wiggle='a', values=[1, 2, 3, 4])


def test_wiggle_parameters_repeated():
    """ Test that wiggle_parameters accepts a nested list as input """
    si.wiggle_parameters(parameters=si.wiggle_parameters(parameters={'a': 1, 'b': 2},
                                                         parameter_to_wiggle='a',
                                                         values=[1, 2, 3, 4]),
                         parameter_to_wiggle='b',
                         values=[10, 20])


def test_wiggle_parameters_parallel():
    """ Test that wiggle_parameters_parallel throws an error if we pass a weird type """
    try:
        output = si.wiggle_parameters_parallel(1, parameter_to_wiggle='a', values=[1, 2, 3, 4])
        raise Exception('This should have failed!')
    except TypeError:
        return


def test_wiggle_parallel():
    """ Test that wiggle_parameters_parallel will accept a list of parameters and values and wiggle them
    together as specified in the docstring """
    output = si.wiggle_parameters_parallel(parameters=[{'a': 1, 'b': 2}],
                                           parameter_to_wiggle=['a', 'b'],
                                           values=[[1, 2], [3, 4]])
    assert output[0][0]['a'] == 1 and output[0][0]['b'] == 3 and output[0][1]['a'] == 2 and output[0][1]['b'] == 4


def test_wiggle_parallel_single_parameter():
    """ Test that wiggle_parameters_parallel will accept a a single parameter and list of values and wiggle them """
    output = si.wiggle_parameters_parallel(parameters=[{'a': 1, 'b': 2}],
                                           parameter_to_wiggle='a',
                                           values=[1, 2])
    assert output[0][0]['a'] == 1 and output[0][0]['b'] == 2 and output[0][1]['a'] == 2 and output[0][1]['b'] == 2


def test_wiggle_parallel_array():
    """ Test that wiggle_parameters_parallel will accept an array of parameters and values and wiggle them
    together as specified in the docstring """
    output = si.wiggle_parameters_parallel(parameters=np.array([{'a': 1, 'b': 2}]),
                                           parameter_to_wiggle=['a', 'b'],
                                           values=[[1, 2], [3, 4]])
    assert output[0]['a'] == 1 and output[0]['b'] == 3 and output[1]['a'] == 2 and output[1]['b'] == 4


def test_wiggle_parallel_error_gen():
    """ Test that wiggle_parameters_parallel will accept a list of parameters and values but raise an error if the
    length of the two lists differs """
    try:
        si.wiggle_parameters_parallel(parameters=[{'a': 1, 'b': 2}],
                             parameter_to_wiggle=['a', 'b'],
                             values=[[1, 2]])
        raise Exception('This should have failed!')
    except ValueError:
        return
