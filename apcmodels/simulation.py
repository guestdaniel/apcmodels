from itertools import product
from pathos.multiprocessing import ProcessPool
from copy import deepcopy


class Simulator:
    """
    Simulator is the core of apcmodels' API for modeling auditory system responses to acoustic stimuli
    """
    def __init__(self, default_runfunc=None):
        if default_runfunc is None:
            self.default_runfunc = self.simulate
        else:
            self.default_runfunc = default_runfunc

    def run(self, batch, runfunc=None, parallel=True, n_thread=8):
        """ Main logical core of Simulator, accepts a function (which defines the simulation and what returns its
        output) and a batch (list with elements to be passed to runfunc), runs the model over the
        sequence, and returns the simulated results as specified in runfunc. In theory, runfunc is allowed to have
        side effects (e.g., saving to disk).

        Arguments:
            batch (list): a list of elements accepted by runfunc as input parameters

            runfunc (func): function that accepts elements of a sequence as arguments and returns simulation results

            parallel (bool): flag to control whether we run the sequence in parallel using pathos multiprocessing or not

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
            results = p.map(runfunc, batch)
        # If not parallel, simply iterate over and run each element of the sequence
        else:
            results = [runfunc(element) for element in batch]
        return results

    def run_batch(self, inputs, input_parameters, model_parameters, mode='product', parallel=True, n_thread=8,
                  runfunc=None):
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

        Returns:
            results (list): list of results
        """
        # If runfunc is None, use default_runfunc
        if runfunc is None:
            runfunc = self.default_runfunc
        # Generate batch
        batch = self.construct_batch(inputs, input_parameters, model_parameters, mode)
        return self.run(batch, runfunc, parallel, n_thread)

    def simulate(self, params):
        """ Dummy method to provide an example runfunc for run() above. Subclasses should implement appropriate
         runfuncs (see run() and run_batch() above). """
        return None

    @staticmethod
    def construct_batch(inputs, input_parameters, model_parameters, mode='product'):
        """
        Combines a set of inputs, a corresponding set of parameters, and set of model parameters into a combined single
        object that can be handled by run() and encodes a series of simulations to run.

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
                temp['input'] = _input
                temp['input_params'] = input_params
                batch.append(temp)
        elif mode == 'product':
            # Loop through product of zipped up combos of elements of inputs and input_parameters with model_parameters
            for _temp, model_params in product(zip(inputs, input_parameters), model_parameters):
                temp = deepcopy(model_params)
                temp['input'] = _temp[0]
                temp['input_params'] = _temp[1]
                batch.append(temp)
        else:
            raise ValueError('Unknown mode!')
        # Return
        return batch


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


def _increment_parameter_callable(paramfunc, inc):
    """ Simple function wrapper to allow us to increment the output of paramfunc() by inc when it is evaluated """
    def inner():
        return paramfunc() + inc
    return inner


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
        # Check if element is callable --- if so, we should apply the increment at evaluation time,
        # otherwise apply now
        if callable(temp[key]):
            temp[key] = _increment_parameter_callable(temp[key], increments[key])
        else:
            temp[key] = temp[key] + increments[key]
        parameter_sequence.append(temp)
    # Return sequence
    return parameter_sequence


def increment_parameters(baselines, increments):
    """
    Takes a dict of parameter names and values (or possibly nested lists of these) and copies its elements multiple
    times. Each copy after the first contains one parameter with a small increment.

    Arguments:
        baselines (dict, list): dict of baseline parameter names and values or a list. If a list, the elements can
            be dicts of parameters or lists. Lists are processed recursively until no lists remain.

        increments (dict): dict of parameter names and values to increment them by

    Returns:
        sequence: list of lists. Elements of lists are dicts. First dict is baselines, remaining dicts contain a
            single incremented parameter (with the size of the increment indicated in increments).
    """
    # Check if input is dict or list and process accordingly
    if type(baselines) is dict:
        return _increment_parameters_dict(baselines, increments)
    elif type(baselines) is list:
        output = [increment_parameters(element, increments) for element in baselines]
        if len(output) == 1:
            return output[0]
        else:
            return output
    else:
        raise ValueError('Input is not list or dict!')


def wiggle_parameters(baselines, parameter, values):
    """
    Takes a dict of parameter names and values (or possibly nested lists of these) and copies its elements multiple
    times. Each copy after the first contains one parameter set to a new parameter value (wiggled).

    Arguments:
        baselines (dict, list): dict of parameter names and values or a list. If a list, the elements can
            be dicts of parameters or lists. Lists are processed recursively until no lists remain.

        parameter (string): name of parameter to wiggle

        values (list): values to which the parameter value is wiggled

    Returns:
        sequence: list of lists. Elements of lists are dicts. First dict is baselines, remaining dicts contain a
            single wiggled parameter.
    """
    # Check if input is dict or list and process accordingly
    if type(baselines) is dict:
        return _wiggle_parameters_dict(baselines, parameter, values)
    elif type(baselines) is list:
        output = [wiggle_parameters(element, parameter, values) for element in baselines]
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


def run_rates_util(params, ratefunc):
    """
    Looks for an input element of params called 'input' and attempts to simulate a response for each element of
    'input' using ratefunc

    Arguments:
        params (dict): a dict whose elements are 'input', containing inputs appropriate for ratefunc, and kwargs to be
            passed to ratefunc

        ratefunc (function): a function that accepts input and other kwargs and returns model simulations

    Returns:
        output: results of applying ratefunc to each input in params

    """
    # If the input is not a list, just run ratefunc
    if type(params['input']) is not list:
        return [ratefunc(**params)]
    # If the input *is* a list, process each input separately
    else:
        output = []
        for _input in params['input']:
            # Make a copy of the parameters and replace the list of inputs with just this current iteration's input
            temp_params = deepcopy(params)
            temp_params['input'] = _input
            output.append(ratefunc(**temp_params))
    return output
