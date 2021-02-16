import apcmodels.simulation as si


def test_parameters_init():
    """ Check to make sure that Parameters.__init__() accepts kwargs that are turned into entries in its params
    attribute """
    params = si.Parameters(hello='world', foo='bar')
    assert params[0]['hello'] == 'world' and params[0]['foo'] == 'bar'


def test_append_parameters_single_element_input():
    """ Check that if we provide append_parameters with an input with just one element that it still returns a list """
    assert type(si.append_parameters([{'hello': 'world'}], 'foo', 'bar')) == list


def test_append_parameters_single_item_to_append():
    """ Check that if we provide append_parameters with an reasonable input (a list of multiple dicts) that it correctly
     appends a parameter to each dict """
    params = [{'x': 1}, {'x': 2}, {'x': 3}]
    params = si.append_parameters(params, 'y', 2)
    assert params[0]['y'] == 2 and params[1]['y'] == 2


def test_append_parameters_multiple_items_to_append():
    """ Check that if we provide append_parameters with an reasonable input (a list of multiple dicts) that it correctly
     appends multiple parameters to each dict """
    params = [{'x': 1}, {'x': 2}, {'x': 3}]
    params = si.append_parameters(params, ['y', 'z'], [2, 3])
    assert params[0]['y'] == 2 and params[1]['y'] == 2 and params[0]['z'] == 3 and params[1]['z'] == 3


def test_append_parameters_bad_inputs1():
    """ Check that if we provide append_parameters with an reasonable input (a list of multiple dicts) but bad inputs
     ( one list and one not list) that it raises the right errors """
    try:
        si.append_parameters({'x': 1}, ['y'], 2)
        raise Exception('This should have failed!')
    except ValueError:
        return


def test_append_parameters_bad_inputs2():
    """ Check that if we provide append_parameters with an reasonable input (a list of multiple dicts) but bad inputs
     (two lists of different lengths) that it raises the right errors """
    try:
        si.append_parameters({'x': 1}, ['y'], [1, 2, 3])
        raise Exception('This should have failed!')
    except ValueError:
        pass


def test_simulator_run_simple_batch():
    """ Check that run can handle arbitrary inputs """
    # Create dummy function that returns input as output --- thus, the output of run() should be identical to its input
    def dummy(params):
        return params['_input']
    # Initialize simulator object
    sim = si.Simulator()
    # Create dummy input
    dummy_input = [{'_input': 1}, {'_input': 2}]
    output = sim.run(params=dummy_input, runfunc=dummy, parallel=True)
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
    """ Check that wiggle_parameters accepts a nested list as input """
    si.wiggle_parameters(parameters=si.wiggle_parameters(parameters={'a': 1, 'b': 2},
                                                         parameter_to_wiggle='a',
                                                         values=[1, 2, 3, 4]),
                         parameter_to_wiggle='b',
                         values=[10, 20])


def test_wiggle_parallel():
    """ Check that wiggle_parameters_parallel will accept a list of parameters and values and wiggle them
    together as specified in the docstring """
    output = si.wiggle_parameters_parallel(parameters=[{'a': 1, 'b': 2}],
                                           parameter_to_wiggle=['a', 'b'],
                                           values=[[1, 2], [3, 4]])
    assert output[0][0]['a'] == 1 and output[0][0]['b'] == 3 and output[0][1]['a'] == 2 and output[0][1]['b'] == 4


def test_wiggle_parallel_error_gen():
    """ Check that wiggle_parameters_parallel will accept a list of parameters and values but raise an error if the
    length of the two lists differs """
    try:
        si.wiggle_parameters_parallel(parameters=[{'a': 1, 'b': 2}],
                             parameter_to_wiggle=['a', 'b'],
                             values=[[1, 2]])
        raise Exception('This should have failed!')
    except ValueError:
        return
