from pathos.multiprocessing import ProcessPool
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import re


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
        return kwargs

    def run(self, params, runfunc=None, parallel=True, n_thread=8, hide_progress=False):
        """ Main logical core of Simulator. run() is designed to be a flexible method that supports running batches
        of simulations using parallelization. run() takes the elements of its only required argument, params, and
        dispatches them to a function (either default or user-provided) either in a loop or using a multiprocessing
        Pool. The elements of params are assumed to be dicts that encode the information required to run simulations or
        to be (possibly nested) lists of such dicts. Each element of params results in a single return and all of these
        returns are bundled and returned either as a list in the same order as params or as an array in the same shape
        as params. In theory, the function used to implement the simulations is allowed to have side effects
        (e.g., saving to disk, writing to a log).

        One disadvantage of the way that run() is currently implemented is that parallelization is only supported
        between elements of params. In other words, only a single core can work on a single element of params. A future
        version plans to provide a way around this.

        Arguments:
            params (list, ndarray): a list or ndarray whose elements are passed to runfunc

            runfunc (func): function that accepts kwargs and returns simulation results

            parallel (bool): flag to control if we run the sequence in parallel using pathos multiprocessing

            n_thread (int): number of threads to use in multiprocessing, ignored if parallel is false

            hide_progress (bool): flag to control if we want to display a tqdm progress bar

        Returns:
            results (list, ndarray): list or ndarray of results
        """
        # If runfunc is None, use the default
        if runfunc is None:
            runfunc = self.default_runfunc
        # If we pass Parameters object, extract underlying data and discard object shell
        if type(params) is Parameters:
            params = params.params
        # If parallel, set up the pool and run sequence on pool
        if parallel:
            p = ProcessPool(n_thread)
            if type(params) is list:
                results = list(p.imap(runfunc, tqdm(params, disable=hide_progress)))
            elif type(params) is np.ndarray:
                # For array params, we need flatten the array and then un-flatten it after output
                old_size = params.shape
                params = np.reshape(params, (params.size,))
                results = list_to_array(list(p.imap(runfunc, tqdm(params, disable=hide_progress))))
                results = np.reshape(results, old_size)
            else:
                raise ValueError('params should be a list or an array')
        # If not parallel, simply iterate over and run each element of the sequence
        else:
            if type(params) is list:
                results = [runfunc(element) for element in params]
            elif type(params) is np.ndarray:
                # For array params, we need flatten the array and then un-flatten it after output
                old_size = params.shape
                params = np.reshape(params, (params.size,))
                results = list_to_array(list(map(runfunc, params)))
                results = np.reshape(results, old_size)
            else:
                raise ValueError('params should be a list or an array')
        return results


class Parameters:
    """
    Parameters provides an object-oriented interface to the xxx_parameters functions below that facilitate the
    construction of parameter lists. The key feature of Parameters() is the params attribute, which encodes a
    series of simulations via an array of dicts (or possibly nested list of dicts). __iter__ and __getitem__ methods
    are implemented to allow users to seamlessly loop or index, and a shape property is implemented to allow the user
    to access the shape of the array directly.
    """
    def __init__(self, **kwargs):
        """
        Arguments:
            **kwargs: when Parameters is initialized, each kwarg passed to __init__ is included as an entry in a seed
            dictionary.
        """
        self.params = np.array([kwargs])

    def __getitem__(self, index):
        return self.params[index]

    def __iter__(self):
        for elem in self.params:
            yield elem

    def __str__(self):
        return self.params.__str__()

    @property
    def shape(self):
        return self.params.shape

    @property
    def size(self):
        return self.params.size

    def append(self, parameter_to_append, value):
        """ See documentation for append_parameters() """
        self.params = append_parameters(self.params, parameter_to_append, value)

    def add_inputs(self, inputs):
        """
        Uses the stitch_parameters() function to add a list of inputs to params with the key '_input'

        Arguments:
            inputs (list, ndarray): list or array of inputs with equal length/shape to self.params.
                Lists will be coerced to arrays.
        """
        self.params = stitch_parameters(self.params, '_input', np.array(inputs, dtype=object))

    def combine(self, parameters_2):
        """ See documentation for combine_parameters() """
        self.params = combine_parameters(self.params, parameters_2)

    def evaluate(self):
        """ See documentation for evaluate_parameters() """
        self.params = evaluate_parameters(self.params)

    def flatten(self):
        """ Takes the params array and flattens it into a single 1d array. We assume that each element of self.params is
         of the same type and of the same structure.
         TODO: add tests?
         """
        # First, we loop through each element of self.params and flatten them out if needed
        temp = np.empty(self.params.shape, dtype=object)
        with np.nditer(self.params, flags=['refs_ok', 'multi_index'], op_flags=['readwrite']) as it:
                for x in it:
                    x = x.item()
                    if type(x) is list:
                        temp[it.multi_index] = flatten_parameters(x)
                    else:
                        temp[it.multi_index] = x
        # Next, we concatenate together all the elements of temp and return them
        self.params = np.concatenate(temp)

    def increment(self, increments):
        """ See documentation for increment_parameters() """
        self.params = increment_parameters(self.params, increments)

    def repeat(self, n_rep):
        """ See documentation for repeat_parameters() """
        self.params = repeat_parameters(self.params, n_rep)

    def stitch(self, parameter_to_stitch, values):
        """ See documentation for stitch_parameters() """
        self.params = stitch_parameters(self.params, parameter_to_stitch, values)

    def wiggle(self, parameter_to_wiggle, values):
        """ See documentation for wiggle_parameters() """
        self.params = wiggle_parameters(self.params, parameter_to_wiggle, values)

    def wiggle_parallel(self, parameter_to_wiggle, values):
        """ See documentation for wiggle_parameters_parallel() """
        self.params = wiggle_parameters_parallel(self.params, parameter_to_wiggle, values)


def append_parameters(parameters, parameter_to_append, value):
    """
    Takes a dict of parameter names and values (or an ndarray of these) and adds the same new key
    and value combo to each dict. Useful for encoding the same information in each element of parameters.

    Arguments:
        parameters (dict, list, ndarray): dict of parameter names and values, or a list or array of such dicts.

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
    elif type(parameters) is np.ndarray:
        temp = np.empty(parameters.shape, dtype=object)
        with np.nditer(parameters, flags=['refs_ok', 'multi_index'], op_flags=['readwrite']) as it:
            for x in it:
                temp[it.multi_index] = append_parameters(x.item(), parameter_to_append, value)
        return temp
    else:
        raise ValueError('Input is not list or dict!')


def combine_parameters(parameters_1, parameters_2):
    """
    Accepts two lists or arrays of dicts of equal length/shape and combines them together. Useful for combining together
    two types of parameter lists (e.g., stimulus and model parameters) of the same length.

    Arguments:
        parameters_1 (list, ndarray): list or ndarray of dicts
        parameters_2 (list, ndarray): list or ndarray of dicts, equal in length/length to parameters_1

    Returns:
        output (list, ndarray): a list or ndarray of dicts, each dict contains all elements from both dicts. Conflicting
         elements (i.e., keys are in both input dicts) are both included by resolved by appending '_2' to the second
         dict's key.
    """
    # Check to make sure that the two lists are the same length
    if type(parameters_1) == list:
        if len(parameters_1) != len(parameters_2):
            raise ValueError('parameters_1 and parameters_2 should be the same length.')
    elif type(parameters_1) == np.ndarray:
        if not np.all(parameters_1.shape == parameters_2.shape):
            raise ValueError('parameters_1 and parameters_2 should be the same shape.')
    # Loop through pairs of elements from parameters_1 and parameters_2, copy parameters_1 element and then add all
    # key-value combos from parameters_2 element
    for params_1, params_2 in zip(parameters_1, parameters_2):
        # Loop through elements of params_2
        for key in params_2.keys():
            if key in params_1.keys():
                params_1[key + '_2'] = params_2[key]
            else:
                params_1[key] = params_2[key]
    # Return
    return parameters_1


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
        parameters (dict, list, ndarray): dict of parameter names and values or a list or ndarray of such dicts. If a
        list, the elements can be dicts of parameters or lists. Lists are processed recursively. If an ndarray, the
        elements can be dicts or lists. Lists are processed recursively.

    Returns:
        parameters (dict, list, ndarray): input but where each callable element in the dict(s) has been evaluated
    """
    if type(parameters) is dict:
        return _evaluate_parameters_dict(parameters)
    elif type(parameters) is list:
        return [evaluate_parameters(element) for element in parameters]
    elif type(parameters) is np.ndarray:
        temp = np.empty(parameters.shape, dtype=object)
        with np.nditer(parameters, flags=['refs_ok', 'multi_index'], op_flags=['readwrite']) as it:
            for x in it:
                temp[it.multi_index] = evaluate_parameters(x.item())
        return temp
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
    Takes a dict of parameter names and values or a (possibly nested) list or an ndarray of such dicts and and copies
    its elements multiple times. Each copy after the first contains one parameter with a small increment. The copies
    are then returned as a list. Useful for setting up simulations for ideal observer analysis.

    Arguments:
        parameters (dict, list, ndarray): dict of baseline parameter names and values or a list or ndarray of such
        dicts. If a list, the elements can be dicts of parameters or lists. Lists are processed recursively. If an
        ndarray, the elements can be dicts or lists. Lists are processed recursively.

        increments (dict): dict of parameter names and values to increment them by

    Returns:
        sequence (list, ndarray): The input, except each dict has been replaced by a list of dicts. First dict
        encodes a baseline while the remaining dicts encode small deviations in a single parameter from that baseline
        (in the same order as specified in increments).
    """
    # Before we increment any parameters, we should evaluate any callable parameters
    parameters = evaluate_parameters(parameters)
    # Check if input is dict or list and process accordingly
    if type(parameters) is dict:
        return _increment_parameters_dict(parameters, increments)
    elif type(parameters) is list:
        return [increment_parameters(element, increments) for element in parameters]
    elif type(parameters) is np.ndarray:
        temp = np.empty(parameters.shape, dtype=object)
        with np.nditer(parameters, flags=['refs_ok', 'multi_index'], op_flags=['readwrite']) as it:
            for x in it:
                temp[it.multi_index] = increment_parameters(x.item(), increments)
        return temp
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
        if len(re.findall('^\(\d+\)', key)) > 0:
            new_key = key[(len(key.split('_')[0])+1):]
            # Create copy of baseline
            temp = deepcopy(baselines)
            # Check if elements of temp are callable, if so we call them now
            if callable(temp[new_key]):
                temp[new_key] = temp[new_key]() + increments[key]
            else:
                temp[new_key] = temp[new_key] + increments[key]
            # Add in a record of the size of the increment
            temp['increment_size'] = increments[key]
            parameter_sequence.append(temp)
        else:
            # Create copy of baseline
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


def repeat_parameters(parameters, n_rep):
    """
    Takes a dict of parameter names and values or a (possibly nested) list or an ndarray of these and replaces each dict
    with a list containing multiple copies of that dict. Useful for encoding repeated simulations at the same parameter
    values.

    Arguments:
        parameters (dict, list, ndarray): dict of parameter names and values or a list. If a list, the elements can
            be dicts of parameters or lists. Lists are processed recursively. If an ndarray, the elements can be dicts
            or lists.

        n_rep (int): number of repetitions to encode by copying the param dicts

    Returns:
        output (list, ndarray): nested lists or ndarray. Each element is corresponds to an element in the input
        parameters but where dicts were replaced with lists containing multiple copies of that dict
    """
    if type(parameters) is dict:
        return _repeat_dict(parameters, n_rep)
    elif type(parameters) is list:
        return [repeat_parameters(element, n_rep) for element in parameters]
    elif type(parameters) is np.ndarray:
        temp = np.empty(parameters.shape, dtype=object)
        with np.nditer(parameters, flags=['refs_ok', 'multi_index'], op_flags=['readwrite']) as it:
            for x in it:
                temp[it.multi_index] = repeat_parameters(x.item(), n_rep)
        return temp
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
    Takes a (possibly nested) list or an array of dicts containing parameter names and values, a name of a new
    parameter, and a list of  values or an ndarray of values to add to each dict. Useful for encoding
    new unique information in each dict.

    Arguments:
        parameters (list, ndarray): list or ndarray of dicts of parameter names and values or lists of such dicts

        parameter_to_stitch (str): new parameter name

        values (list, ndarray): list where each element is a value for the new parameter to be set to, or a nested list
            of such lists. The length/hierarchy of parameters and values must match exactly. Alternatively, it can be
            an ndarray of the same shape as the input

    Returns:
        output (list, ndarray): input except with parameter_to_stitch encoded in each dict
    """
    # Check to make sure that the two lists are the same length
    if type(parameters) is list:
        if len(parameters) != len(values):
            raise ValueError('parameters and values should be the same length.')
    elif type(parameters) is np.ndarray or type(parameters) is Parameters:
        if not np.all(parameters.shape == values.shape):
            raise ValueError('parameters and values should be the same shape.')
    # Add values to each element in parameters
    if type(parameters) is list:
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
    elif type(parameters) is np.ndarray:
        output = np.empty(parameters.shape, dtype=object)
        with np.nditer(parameters, flags=['refs_ok', 'multi_index'], op_flags=['readwrite']) as it:
            for x in it:
                x = x.item()
                if type(x) is list and type(values[it.multi_index]) is list:
                    output[it.multi_index] = stitch_parameters(x, parameter_to_stitch, values[it.multi_index])
                else:
                    output[it.multi_index] = dict(**x, **{parameter_to_stitch: values[it.multi_index]})
        return output
    else:
        raise ValueError('parameters is of unexpected type')


def wiggle_parameters(parameters, parameter_to_wiggle, values):
    """
    Takes a dict of parameter names and values (or possibly nested lists or arrays of these) and copies its elements
    multiple times. Each copy after the first contains one parameter set to a new parameter value (wiggled). The copies
    are returned. Useful for constructing batches of simulations over a range of parameter values.

    Arguments:
        parameters (dict, list, ndarray): dict of parameter names and values or a list or ndarray of such dicts. If a
        list, the elements can be dicts of parameters or lists. Lists are processed recursively. If an ndarray, the
        elements can be dicts or lists. Lists are processed recursively. Wiggled parameters are added as new dimensions
        to the ndarray, which is squeezed on output to avoid singleton dimensions. This functionality is useful to
        encode a series of wiggled parameters as a multi-dimensional array.

        parameter_to_wiggle (string): name of parameter to wiggle

        values (list, ndarray): values to which the parameter value is wiggled

    Returns:
        sequence: list of lists. Elements of lists are dicts. First dict is baselines, remaining dicts contain a
            single wiggled parameter.

    Notes:
        Generally, we assume that wiggling will be done before any repeats/increments... however, wiggle_parameters() is
        written in an agnostic form such that this assumption is not rigidly enforced. A future version may avoid this
        as it can produce undesireable functionality. # TODO: address this issue
    """
    # Check if input is dict or list and process accordingly
    if type(parameters) is dict:
        return _wiggle_parameters_dict(parameters, parameter_to_wiggle, values)
    elif type(parameters) is list:
        return [wiggle_parameters(element, parameter_to_wiggle, values) for element in parameters]
    elif type(parameters) is np.ndarray:
        return np.squeeze(np.array(list(map(lambda x: wiggle_parameters(x, parameter_to_wiggle, values), parameters))))
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
    Takes a dict of parameter names and values (or possibly nested lists or arrays of these) and copies its elements
    multiple times. Each copy after the first contains one parameter set to a new parameter value (wiggled). The copies
    are returned. Useful for constructing batches of simulations over a range of parameter values. This function differs
    from wiggle_parameters() in that multiple parameter names (and corresponding sets of values) can be specified. In
    this case, these values are wiggled "simultaneously", meaning that each copied dict will simultaneously have adjusted
    all parameters specified by the user.

    Arguments:
        parameters (dict, list, ndarray): dict of parameter names and values or a list or ndarray of such dicts. If a
            list, the elements can be dicts of parameters or lists. Lists are processed recursively. If an ndarray, the
            elements can be dicts or lists. Lists are processed recursively. Wiggled parameters are added as new
            dimensions to the ndarray, which is squeezed on output to avoid singleton dimensions. This functionality is
            useful to encode a series of wiggled parameters as a multi-dimensional array.

        parameter_to_wiggle (string, list): name of parameter to wiggle, or list of such names. If a list, its length
            should match that of values.

        values (list, ndarray): values to which the parameter value is wiggled, or list of such lists. If a list, its
            length should match that of parameters_to_wiggle. The lists can also be 1d arrays.

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
    elif type(parameters) is np.ndarray:
        return np.squeeze(np.array(list(map(lambda x: wiggle_parameters_parallel(x, parameter_to_wiggle, values), parameters))))
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


def list_to_array(_list):
    """
    Accepts a list and turns it into an ndarray with 'object' dtype of shape (len,)

    Arguments:
        _list (list)
    """
    new_array = np.empty(shape=(len(_list), ), dtype='object')
    for ii in range(len(_list)):
        new_array[ii] = _list[ii]
    return new_array
