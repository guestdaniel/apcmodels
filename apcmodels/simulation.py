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

    def simulate(self, **kwargs):
        """ Dummy method to provide an example runfunc for run() above. Subclasses should implement appropriate
         runfuncs (see run() and run_batch() below). """
        return None

    def run(self, batch, runfunc=None, parallel=True, n_thread=8):
        """ Main logical core of Simulator. run() is designed to be a flexible method that supports running batches
        of simulations using parallelization. run() takes the elements of its only required argument, batch, and
        dispatches them to a function (either default or user-provided) either in a loop or using a multiprocessing
        Pool. The elements of batch are assumed to be dicts that encode the information required to run simulations and
        these dicts are unpacked so their elements can be passed as kwargs to the simulation function. Each dict
        in batch results in a single return and all of these returns are bundled and returned as a list in the same
        order as batch. In theory, the function used to implement the simulations is allowed to have side effects (e.g.,
        saving to disk, writing to a log).

        One disadvantage of the way that run() is currently implemented is that parallelization is only supported
        between elements of batch. In other words, only a single core can work on a single element of batch.

        Arguments:
            batch (list): a list of dicts whose elements are passed to runfunc as kwargs

            runfunc (func): function that accepts kwargs and returns simulation results

            parallel (bool): flag to control if we run the sequence in parallel using pathos multiprocessing

            n_thread (int): number of threads to use in multiprocessing, ignored if parallel is false

        Returns:
            results (list): list of results
        """
        # If runfunc is None, use the default
        if runfunc is None:
            runfunc = self.default_runfunc

        # Now, wrap runfunc around a function that takes the elements of batch and unpacks them
        def runfunc_wrapper(kwargs):
            return runfunc(**kwargs)

        # If parallel, set up the pool and run sequence on pool
        if parallel:
            p = ProcessPool(n_thread)
            results = p.map(runfunc_wrapper, batch)
        # If not parallel, simply iterate over and run each element of the sequence
        else:
            results = [runfunc_wrapper(element) for element in batch]
        return results

    def run_batch(self, inputs, input_parameters, model_parameters, mode='product', parallel=True, n_thread=8,
                  runfunc=None, parameters_to_append=None):
        """
        Combines a set of inputs, a corresponding set of parameters, and set of model parameters into a combined single
        object encoding a series of simulations to be run. Runs the simulations and returns the results.

        Arguments:
            inputs (list): list of inputs to run the model on. The elements of inputs can be virtually anything.

            input_parameters (list): list of parameters corresponding to inputs. The elements of input_parameters can
                be virtually anything, but if they are dicts they are unpacked.

            model_parameters (list): list of parameter values to run the model at.

            mode (string): controls how the input and parameter sequences are combined. In 'zip' mode,
                parameter_sequence and input_sequence are combined into a unified sequence element-by-element, such
                that the first element of input_sequence will be run at the parameter values specified in the first
                element of parameter_sequence. In 'product' mode, every element of input_sequence will be run at
                every element of parameter sequence.

            parallel (bool): flag to control whether we run the sequence in parallel using pathos multiprocessing or not

            n_thread (int): number of threads to use in multiprocessing, ignored if parallel is false

            runfunc (func): function that accepts elements of a sequence as arguments and returns simulation results

            parameters_to_append (dict): a dict of parameter names and values, each is appended to batch using
                append_parameters

        Returns:
            results (list): list of results
        """
        # If runfunc is None, use default_runfunc
        if runfunc is None:
            runfunc = self.default_runfunc
        # Generate batch
        batch = self.construct_batch(inputs, input_parameters, model_parameters, mode)
        # Append any parameters
        if parameters_to_append is not None:
            for key in parameters_to_append.keys():
                batch = append_parameters(batch, key, parameters_to_append[key])
        return self.run(batch, runfunc, parallel, n_thread)

    @staticmethod
    def construct_batch(inputs, input_parameters, model_parameters, mode='product'):
        """
        Combines a set of inputs, a corresponding set of input parameters, and set of model parameters into a combined
        single object that encodes a series of simulations to run and is appropriate to pass to run()

        Arguments:
            inputs (list): list of inputs to run the model on. The elements of inputs can be virtually anything.
            
            input_parameters (list): list of parameters corresponding to inputs. The elements of input_parameters can 
                be virtually anything, but if they are dicts they are unpacked. 

            model_parameters (list): list of parameter values to run the model at. 

            mode (string): controls how the input and parameter sequences are combined. In 'zip' mode,
                parameter_sequence and input_sequence are combined into a unified sequence element-by-element, such
                that the first element of input_sequence will be run at the parameter values specified in the first
                element of parameter_sequence. In 'product' mode, every element of input_sequence will be run at
                every element of parameter sequence.

        Returns:
            batch: list of dicts containing (1) model parameters, (2) inputs, and (3) corresponding input parameters.
                All of these are stored as elements of the dicts.
        """
        # Check that input_sequence and input_parameter_sequence are the same length
        assert len(inputs) == len(input_parameters)
        # Create empty list that will contain a master sequence of combined stimuli and parameters
        batch = []
        # Fork based on whether we're going to 'zip' or 'permute'
        if mode == 'zip':
            # Since we're in zip mode, check to make sure all of the inputs have the same length
            assert len(inputs) == len(model_parameters)
            # Now, loop through zipped up combos of elements of inputs, input_parameters, and model_parameters
            for _input, input_params, model_params in zip(inputs, input_parameters, model_parameters):
                temp = deepcopy(model_params)
                temp['_input'] = _input
                temp['input_params'] = input_params
                batch.append(temp)
        elif mode == 'product':
            # Loop through product of zipped up combos of elements of inputs and input_parameters with model_parameters
            for _temp, model_params in product(zip(inputs, input_parameters), model_parameters):
                temp = deepcopy(model_params)
                temp['_input'] = _temp[0]
                temp['input_params'] = _temp[1]
                batch.append(temp)
        else:
            raise ValueError('Unknown mode!')
        # Return
        return batch


def append_parameters(parameters, parameter_to_append, value):
    """
    Takes a dict of parameter names and values (or possibly nested lists of these) and adds a new key and value combo
    to each dict

    Arguments:
        parameters (dict, list): dict of parameter names and values or a list. If a list, the elements can
            be dicts of parameters or lists. Lists are processed recursively until no lists remain.

        parameter_to_append (string): name of parameter to add to each param dict

        value: value to set new parameter to

    Returns:
        parameters: updated parameters
    """
    # Check if input is dict or list and process accordingly
    if type(parameters) is dict:
        parameters[parameter_to_append] = value
        return parameters
    elif type(parameters) is list:
        output = [append_parameters(element, parameter_to_append, value) for element in parameters]
        if len(output) == 1:
            return output[0]
        else:
            return output
    else:
        raise ValueError('Input is not list or dict!')


def flatten_parameters(parameters):
    """
    Takes a (possibly nested) list of parameter dicts and flattens it into a single list

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
        output = [evaluate_parameters(element) for element in parameters]
        if len(output) == 1:
            return output[0]
        else:
            return output
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
    times. Each copy after the first contains one parameter with a small increment.

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
        output = [increment_parameters(element, increments) for element in parameters]
        if len(output) == 1:
            return output[0]
        else:
            return output
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
    containing multiple copies of that dict

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
        output = [repeat(element, n_rep) for element in parameters]
        if len(output) == 1:
            return output[0]
        else:
            return output
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


def wiggle_parameters(parameters, parameter_to_wiggle, values):
    """
    Takes a dict of parameter names and values (or possibly nested lists of these) and copies its elements multiple
    times. Each copy after the first contains one parameter set to a new parameter value (wiggled).

    Arguments:
        parameters (dict, list): dict of parameter names and values or a list. If a list, the elements can
            be dicts of parameters or lists. Lists are processed recursively until no lists remain.

        parameter_to_wiggle (string): name of parameter to wiggle

        values (list): values to which the parameter value is wiggled

    Returns:
        sequence: list of lists. Elements of lists are dicts. First dict is baselines, remaining dicts contain a
            single wiggled parameter.
    """
    # Check if input is dict or list and process accordingly
    if type(parameters) is dict:
        return _wiggle_parameters_dict(parameters, parameter_to_wiggle, values)
    elif type(parameters) is list:
        output = [wiggle_parameters(element, parameter_to_wiggle, values) for element in parameters]
        if len(output) == 1:
            return output[0]
        else:
            return output
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
