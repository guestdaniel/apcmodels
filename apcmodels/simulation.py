from copy import deepcopy
from itertools import product
from pathos.multiprocessing import ProcessPool

class Simulator:
    """
    Simulator is the core of apcmodels' API for modeling auditory system responses to acoustic stimuli
    """
    def __init__(self):
        self.model = None

    def run(self, runfunc, batch, parallel=True, n_thread=8):
        """ Main logical core of Simulator, accepts a function (which defines the simulation and what returns it
        provides) and a sequence (list of dicts containing parameter names and values), runs the model over the
        sequence, and returns the simulated results as specified in runfunc. In theory, runfunc is allowed to have
        side effects (e.g., saving to disk).

        Arguments:
            runfunc (func): function that accepts elements of a sequence as arguments and returns simulation results

            batch (list): a list of elements accepted by runfunc as input parameters, generally produced by
                construct_batch for the runfuncs provided by apcmodels.

            parallel (bool): flag to control whether we run the sequence in parallel using pathos multiprocessing or not

            n_thread (int): number of threads to use in multiprocessing, ignored if parallel is false

        Returns:
            results (list): list of results
        """
        # If parallel, set up the pool and run sequence on pool
        if parallel:
            p = ProcessPool(n_thread)
            results = p.map(runfunc, batch)
        # If not parallel, simply iterate over and run each element of the sequence
        else:
            results = [runfunc(element) for element in batch]
        return results

    def simulate(self, params):
        """ Dummy method to provide an example runfunc for run() above. Subclasses should implement appropriate
         runfuncs (e.g., simulate_firing_rate). """
        return None

    @staticmethod
    def construct_batch(input_sequence, input_parameter_sequence, parameter_sequence, mode='zip'):
        """
        Combines an input sequence, an input parameter sequence, and a model parameter sequence into a combined sequence
        that can be handled by run() to run a series of simulations.

        Arguments:
            input_sequence (list): list containing inputs to run the model on.

            input_parameter_sequence (list): list containing input parameters corresponding to the inputs in
                input_sequence.

            parameter_sequence (list): list of parameter values to run the model at. Each element of the list is a dict
                of parameter names and values that match the expected keyword arguments in run()

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
        assert len(input_sequence) == len(input_parameter_sequence)
        # Create empty list that will contain a master sequence of combined stimuli and parameters
        batch = []
        # Fork based on whether we're going to 'zip' or 'permute'
        if mode == 'zip':
            for input, params in zip(zip(input_sequence, input_parameter_sequence), parameter_sequence):
                params['input'] = input[0]
                params['input_params'] = input[1]
                batch.append(params)
        elif mode == 'product':
            for input, params in product(zip(input_sequence, input_parameter_sequence), parameter_sequence):
                params['input'] = input[0]
                params['input_params'] = input[1]
                batch.append(params)
        else:
            raise ValueError('Unknown mode!')
        # Return
        return batch


class Synthesizer:
    """
    Synthesizer defines a general framework for defining and using a synthesis routine for a stimulus

    Arguments:
        stimulus_name (string): human-readable name of the stimulus type
    """
    def __init__(self, stimulus_name='none'):
        self.stimulus_name = stimulus_name

    def synthesize(self, **kwargs):
        """
        Method to synthesize the stimulus at set parameter values, implemented by subclasses
        """
        return None

    @staticmethod
    def create_incremented_parameter_sequence(baselines, increments):
        """
        Creates a parameter sequence that would be accepted by synthesize_parameter_sequence (i.e., a list of dicts
        specifying parameters and their values). This sequence begins with a baseline set of parameter values and then
        each successive element increments a single parameter by the corresponding value specified in increments (while
        all other parameters are restored to their baseline values).

        Arguments:
            baselines (dict): dict of baseline parameter names and values

            increments (dict): dict of parameter names and values to increment them by

        Returns:
            sequence: list of dicts to be passed to synthesize_parameter_sequence()
        """
        # Check that baselines and increments has the same number of parameters
        assert len(baselines) == len(increments)
        # Create empty storage for output
        parameter_sequence = list()
        # Append a baseline parameter vector to sequence
        parameter_sequence.append(deepcopy(baselines))
        # Loop through elements of increments and construct a corresponding entry in sequence
        for key in increments.keys():
            temp = deepcopy(baselines)
            temp[key] = temp[key] + increments[key]
            parameter_sequence.append(temp)
        # Return sequence
        return parameter_sequence

    @staticmethod
    def wiggle_parameter(paramdict, parameter):
        """
        Takes a parameter

        Arguments:
            baselines (dict): dict of baseline parameter names and values

            increments (dict): dict of parameter names and values to increment them by

        Returns:
            sequence: list of dicts to be passed to synthesize_parameter_sequence()
        """

        TODOODODODODOD

    def apply_increment_to_sequence(self, baseline_sequence, increments):
        """
        Takes a parameter sequence and transforms each element into an incremented parameter sequence

        Arguments:
            baseline_sequence (list): list of dicts of baseline parameter names and values

            increments (dict): dict of parameter names and values to increment them by

        Returns:
            new_sequence: list of lists of dicts to be passed to synthesize_parameter_sequence() # TODO: adjust

        # TODO: add tests if needed
        """
        return [self.create_incremented_parameter_sequence(baselines, increments) for baselines in baseline_sequence]

    def synthesize_parameter_sequence(self, parameter_sequence, **kwargs):
        """
        Repeatedly synthesizes copies of the stimulus based on the sequence of parameters provided by
        parameter_sequence. kwargs allows the user to pass additional keyword arguments to synthesize()

        Arguments:
            parameter_sequence (list): list of parameter values to synthesize. Each element of the list is either a dict
                of parameter names and values that match the expected keyword arguments in synthesize() or a list
                itself. If the element is a list, synthesize_parameter_sequence is applied recursively until no lists
                remain.

        Returns:
            results: list of ndarrays containing synthesized copies of the stimulus
        """
        # Create empty storage for synthesized copies of stimulus
        results = []
        # Loop through the parameter sequence and pass it to synthesize(), unpacking the parameter dict in place
        for parameter_vector in parameter_sequence:
            if type(parameter_vector) is list:
                results.append(self.synthesize_parameter_sequence(parameter_vector, **kwargs))
            else:
                results.append(self.synthesize(**parameter_vector, **kwargs))
        return results


def simulate_firing_rates(model, paramdict):
    """
    Simulates a response for a model neuron and returns that neuron's instantaneous firing rate

    Arguments:
        model: an function that accepts input ndarrays and **kwargs and returns a firing rate ndarray

        paramdict: a dict containing at least an element called 'input' containing an input and also optionally various
            parameter names and values. They are unpacked and passed to the model as keyword arguments.

    Returns:
        output: an output ndarray of firing rates

    """
    return model(**paramdict)


