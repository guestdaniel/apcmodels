from itertools import product
from pathos.multiprocessing import ProcessPool
from copy import deepcopy


class Simulator:
    """
    Simulator is the core of apcmodels' API for modeling auditory system responses and decoding or processing the
    outputs of the models.
    """
    def __init__(self, default_runfunc=None):
        if default_runfunc is None:
            self.default_runfunc = self.simulate
        else:
            self.default_runfunc = default_runfunc

    def simulate(self, kwargs):
        """ Dummy method to provide an example runfunc for run() above. Subclasses should implement appropriate
         runfuncs (see run() and run_batch() below). """
        return None

    def run(self, params, runfunc=None, parallel=True, n_thread=8):
        """ Main logical core of Simulator. run() is designed to be a flexible method that supports running batches
        of simulations using parallelization. run() takes the elements of its only required argument, params, and
        dispatches them to a function (either default or user-provided) either in a loop or using a multiprocessing
        Pool. The elements of params are assumed to be dicts that encode the information required to run simulations and
        these dicts are unpacked so their elements can be passed as kwargs to the simulation function. Each dict
        in params results in a single return and all of these returns are bundled and returned as a list in the same
        order as params. In theory, the function used to implement the simulations is allowed to have side effects
        (e.g., saving to disk, writing to a log).

        One disadvantage of the way that run() is currently implemented is that parallelization is only supported
        between elements of params. In other words, only a single core can work on a single element of params. A future
        version plans to provide a way around this.

        Arguments:
            params (list): a list of dicts whose elements are passed to runfunc as kwargs

            runfunc (func): function that accepts kwargs and returns simulation results

            parallel (bool): flag to control if we run the sequence in parallel using pathos multiprocessing

            n_thread (int): number of threads to use in multiprocessing, ignored if parallel is false

        Returns:
            results (list): list of results
        """
        # If runfunc is None, use the default
        if runfunc is None:
            runfunc = self.default_runfunc

        # If parallel, set up the pool and run sequence on pool
        if parallel:
            p = ProcessPool(n_thread)
            results = p.map(runfunc, params)
        # If not parallel, simply iterate over and run each element of the sequence
        else:
            results = [runfunc(element) for element in params]
        return results


def append_parameters(parameters, parameter_to_append, value):
    """
    Takes a dict of parameter names and values (or possibly nested lists of these) and adds the same new key and value
    combo to each dict. Useful for encoding the same information in each element of parameters.

    Arguments:
        parameters (dict, list): dict of parameter names and values or a list. If a list, the elements can
            be dicts of parameters or lists. Lists are processed recursively until no lists remain.

        parameter_to_append (string, list): name of parameter to add to each param dict, or a list of such names

        value: value to set new parameter to, or a list of such values

    Returns:
        parameters: updated parameters
    """
    # Check if input is dict or list and process accordingly
    if type(parameters) is dict:
        if type(parameter_to_append) is list:
            if type(value) is not list:
                raise ValueError('value argument should be a list because parameter_to_append is list')
            if len(parameter_to_append) != len(value):
                raise ValueError('len of value and parameter_to_append should be equal')
            for _param, _val in zip(parameter_to_append, value):
                parameters[_param] = _val
        else:
            parameters[parameter_to_append] = value
        return parameters
    elif type(parameters) is list:
        return [append_parameters(element, parameter_to_append, value) for element in parameters]
    else:
        raise ValueError('Input is not list or dict!')


def combine_parameters(parameters_1, parameters_2):
    """
    Accepts two lists of dicts of equal length and combines them together. Useful for combining together two types of
    parameter lists (e.g., stimulus and model parameters) of the same length.

    Arguments:
        parameters_1 (list): list of dicts
        parameters_2 (list): list of dicts, equal in length to parameters_1

    Returns:
        output (list): a list of dicts, each dict contains all elements from both dicts. Conflicting elements (i.e.,
            keys are in both input dicts) are both included by resolved by appending '_2' to the second dict's key.
    """
    # Check to make sure that the two lists are the same length
    if len(parameters_1) != len(parameters_2):
        raise ValueError('parameters_1 and parameters_2 should be the same length.')
    # Create empty output list
    output = list()
    # Loop through pairs of elements from parameters_1 and parameters_2, copy parameters_1 element and then add all
    # key-value combos from parameters_2 element
    for params_1, params_2 in zip(parameters_1, parameters_2):
        # Loop through elements of params_2
        for key in params_2.keys():
            if key in params_1.keys():
                params_1[key + '_2'] = params_2[key]
            else:
                params_1[key] = params_2[key]
        output.append(params_1)
    # Return
    return output


def flatten_parameters(parameters):
    """
    Takes a (possibly nested) list of parameter dicts and flattens it into a single list.

    Arguments:
        parameters (list): possibly nested list of parameters dicts

    Returns:
        flat_parameters (list): flattened list of parameter dicts
    """
    if parameters == []:
        return parameters
    if isinstance(parameters[0], list):
        return flatten_parameters(parameters[0]) + flatten_parameters(parameters[1:])
    return parameters[:1] + flatten_parameters(parameters[1:])


def evaluate_parameters(parameters):
    """
    Takes a dict of parameter names and values or a (possibly nested) list of these and evaluates any callable elements
    in the dicts.

    Arguments:
        parameters (dict, list): dict of parameter names and values or a list. If a list, the elements can
            be dicts of parameters or lists. Lists are processed recursively until no lists remain.

    Returns:
        parameters (dict, list): input but where each callable element in the dict(s) has been evaluated
    """
    if type(parameters) is dict:
        return _evaluate_parameters_dict(parameters)
    elif type(parameters) is list:
        return [evaluate_parameters(element) for element in parameters]
    else:
        raise ValueError('Input is not list or dict!')


def _evaluate_parameters_dict(paramdict):
    """
    Takes a dict of parameter names and values and evaluates any callable elements in the dicts.

    Arguments:
        paramdict (dict): dict of parameter names and values

    Returns:
        paramdict (dict): input but where each callable element in the dict has been evaluated
    """
    temp = deepcopy(paramdict)
    # Check if elements of temp are callable, if so we call them now
    for key in temp.keys():
        if callable(temp[key]):
            temp[key] = temp[key]()
    return temp


def increment_parameters(parameters, increments):
    """
    Takes a dict of parameter names and values (or possibly nested lists of these) and copies its elements multiple
    times. Each copy after the first contains one parameter with a small increment. Useful for setting up simulations
    for ideal observer analysis.

    Arguments:
        parameters (dict, list): dict of baseline parameter names and values or a list. If a list, the elements can
            be dicts of parameters or lists. Lists are processed recursively until no lists remain.

        increments (dict): dict of parameter names and values to increment them by

    Returns:
        sequence: list of lists. Elements of lists are dicts. First dict is baselines, remaining dicts contain a
            single incremented parameter (with the size of the increment indicated in increments).
    """
    # Before we increment any parameters, we should evaluate any callable parameters
    parameters = evaluate_parameters(parameters)
    # Check if input is dict or list and process accordingly
    if type(parameters) is dict:
        return _increment_parameters_dict(parameters, increments)
    elif type(parameters) is list:
        return [increment_parameters(element, increments) for element in parameters]
    else:
        raise ValueError('Input is not list or dict!')


def _increment_parameters_dict(baselines, increments):
    """
    Takes a dict of parameter names and values and copies it multiple times. Each copy after the first contains
    one incremented parameter.

    Arguments:
        baselines (dict): dict of baseline parameter names and values

        increments (dict): dict of parameter names and values to increment them by

    Returns:
        sequence: list of dicts. First dict is baselines, remaining dicts contain a
            single incremented parameter (with the size of the increment indicated in increments).
    """
    # Create empty storage for output
    parameter_sequence = list()
    # Append a baseline parameter vector to sequence
    parameter_sequence.append(deepcopy(baselines))
    # Loop through elements of increments and construct a corresponding entry in sequence
    for key in increments.keys():
        temp = deepcopy(baselines)
        # Check if elements of temp are callable, if so we call them now
        if callable(temp[key]):
            temp[key] = temp[key]() + increments[key]
        else:
            temp[key] = temp[key] + increments[key]
        # Add in a record of the size of the increment
        temp['increment_size'] = increments[key]
        parameter_sequence.append(temp)
    # Return sequence
    return parameter_sequence


def repeat(parameters, n_rep):
    """
    Takes a dict of parameter names and values or a (possibly nested) list of these and replaces each dict with a list
    containing multiple copies of that dict. Useful for encoding repeated simulations at the same parameter values.

    Arguments:
        parameters (dict, list): dict of parameter names and values or a list. If a list, the elements can
            be dicts of parameters or lists. Lists are processed recursively until no lists remain.

        n_rep (int): number of repetitions to encode by copying the param dicts

    Returns:
        output (list): nested lists. Each element is corresponds to an element in the input parameters but is replaced
             with a list containing multiple copies of that input
    """
    if type(parameters) is dict:
        return _repeat_dict(parameters, n_rep)
    elif type(parameters) is list:
        return [repeat(element, n_rep) for element in parameters]
    else:
        raise ValueError('Input is not list or dict!')


def _repeat_dict(paramdict, n_rep):
    """
    Takes a dict of parameter names and values and returns a list containing multiple copies of the dict

    Arguments:
        paramdict (dict): dict of parameter names and values

        n_rep (int): number of repetitions to encode by copying the param dicts

    Returns:
        output (list): List containing multiple copies of paramdict
    """
    return [deepcopy(paramdict) for rep in range(n_rep)]


def stitch_parameters(parameters, parameter_to_stitch, values):
    """
    Takes a (possibly nested) list of parameter names and values, a name of a new parameter, and a (possible nested)
    list of values for that parameter to add to each corresponding element of the parameter list. Useful for encoding
    new unique information in each element of parameters.

    Arguments:
        parameters (list): list of dicts of parameter names and values, or a nested list of such lists

        parameter_to_stitch (str): new parameter name

        values (list): list where each element is a value for the new parameter to be set to, or a nested list of such
            lists. The length/hierarchy of parameters and values must match exactly.

    Returns:
        output (list): list of dicts of parameter names and values
    """
    # Check to make sure inputs are the same length
    if len(parameters) != len(values):
        raise ValueError('parameters and values should be the same length')
    # Add values to each element in parameters
    output = list()
    for paramdict, val in zip(parameters, values):
        # Check to see what types we're handling
        if type(paramdict) is list and type(val) is list:
            output.append(stitch_parameters(paramdict, parameter_to_stitch, val))
        else:
            paramdict[parameter_to_stitch] = val
            output.append(paramdict)
    # Return
    return output


def wiggle_parameters(parameters, parameter_to_wiggle, values):
    """
    Takes a dict of parameter names and values (or possibly nested lists of these) and copies its elements multiple
    times. Each copy after the first contains one parameter set to a new parameter value (wiggled). Useful for
    constructing batches of simulations over a range of parameter values.

    Arguments:
        parameters (dict, list): dict of parameter names and values or a list. If a list, the elements can
            be dicts of parameters or lists. Lists are processed recursively until no lists remain.

        parameter_to_wiggle (string): name of parameter to wiggle

        values (list, ndarray): values to which the parameter value is wiggled

    Returns:
        sequence: list of lists. Elements of lists are dicts. First dict is baselines, remaining dicts contain a
            single wiggled parameter.
    """
    # Check if input is dict or list and process accordingly
    if type(parameters) is dict:
        return _wiggle_parameters_dict(parameters, parameter_to_wiggle, values)
    elif type(parameters) is list:
        return [wiggle_parameters(element, parameter_to_wiggle, values) for element in parameters]
    else:
        raise ValueError('Input is not list or dict!')


def _wiggle_parameters_dict(paramdict, parameter, values):
    """
    Takes a dict of parameter names and values and copies it multiple times. Each copy after the first contains
    one parameter set to a new value (wiggled).

    Arguments:
        paramdict (dict): dict of baseline parameter names and values

        parameter (string): name of parameter to wiggle

        values (list): values to which the parameter value is wiggled

    Returns:
        sequence: list of dicts. Each element of the dict corresponds to one value of values
    """
    # Create emtpy storage for output
    parameter_sequence = list()
    # Loop through values and construct a corresponding entry in sequence
    for val in values:
        temp = deepcopy(paramdict)
        temp[parameter] = val
        parameter_sequence.append(temp)
    # Return sequence
    return parameter_sequence


def wiggle_parameters_parallel(parameters, parameter_to_wiggle, values):
    """
    Takes a dict of parameter names and values (or possibly nested lists of these) and copies its elements multiple
    times. Each copy after the first contains parameters set to new values (wiggled). This function differs from
    wiggle_parameters() in that multiple parameter names (and corresponding sets of values) can be specified. In this
    case, these values are wiggled "simultaneously", meaning that each copied dict will simultaneously have adjusted
    all parameters specified by the user.

    Arguments:
        parameters (dict, list): dict of parameter names and values or a list. If a list, the elements can
            be dicts of parameters or lists. Lists are processed recursively until no lists remain.

        parameter_to_wiggle (string, list): name of parameter to wiggle, or list of such names. If a list, its length
            should match that of values.

        values (list): values to which the parameter value is wiggled, or list of such lists. If a list, its length
            should match that of parameters_to_wiggle.

    Returns:
        sequence: list of lists. Elements of lists are dicts. First dict is baselines, remaining dicts contain a
            single wiggled parameter.
    """
    # Check if parameters_to_wiggle is a string or a list and handle accordingly
    if type(parameter_to_wiggle) is list:
        if len(parameter_to_wiggle) != len(values):
            raise ValueError('parameters_to_wiggle and values should have the same length!')
    # Check if input is dict or list and process accordingly
    if type(parameters) is dict:
        return _wiggle_parameters_parallel_dict(parameters, parameter_to_wiggle, values)
    elif type(parameters) is list:
        return [wiggle_parameters_parallel(element, parameter_to_wiggle, values) for element in parameters]
    else:
        raise ValueError('Input is not list or dict!')


def _wiggle_parameters_parallel_dict(paramdict, parameter, values):
    """
    Takes a dict of parameter names and values and copies it multiple times. Each copy after the first contains
    one or multiple parameters set to new values (wiggled).

    Arguments:
        paramdict (dict): dict of baseline parameter names and values

        parameter (string, list): name of parameter to wiggle, or list of such names. If a list, its length
            should match that of values.

        values (list): values to which the parameter value is wiggled, or list of such lists. If a list, its length
            should match that of parameters_to_wiggle.

    Returns:
        sequence: list of dicts
    """
    # Create emtpy storage for output
    parameter_sequence = list()
    # Check if parameter is a string or a list
    if type(parameter) is str:
        # Loop through values and construct a corresponding entry in sequence
        for val in values:
            temp = deepcopy(paramdict)
            temp[parameter] = val
            parameter_sequence.append(temp)
    else:
        # First, we construct transform values from a list of lists into a list of tuples where each tuple contains
        # one element from each list in values
        values = zip(*values)
        # Second, we loop through all these tuples and create a copy of paramdict, update it with new parameter values,
        # and then append it to parameter_sequence
        for valtuple in values:
            temp = deepcopy(paramdict)
            for param_name, param_val in zip(parameter, valtuple):
                temp[param_name] = param_val
            parameter_sequence.append(temp)
    # Return sequence
    return parameter_sequence