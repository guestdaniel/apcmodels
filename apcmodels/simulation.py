from copy import deepcopy

class Simulator:
    """
    Simulator is the core of apcmodels' API for modeling auditory system responses to acoustic stimuli
    """
    def __init__(self):
        self.model = None
        self.synthesizer = None
        self.result = None
        self.queue = Queue()

    def run(self):
        return

    def _print_simulation_info(self):
        print(self.model)
        print(self.synthesizer)


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

    def create_incremented_sequence(self, baselines, increments):
        """
        Creates a parameter sequence that would be accepted by run_sequence (i.e., a list of dicts specifying parameters
        and their values). This sequence begins with a baseline set of parameter values and then each successive element
        increments a single parameter by the corresponding value specified in increments (while all other parameters are
        restored to their baseline values).

        Arguments:
            baselines (dict): dict of baseline parameter values
            increments (dict): dict of parameter names and values to increment them by

        Returns:
            sequence: list of dicts to be passed to run_sequence()
        """
        # Check that baselines and increments has the same number of parameters
        assert len(baselines) == len(increments)
        # Create empty storage for output
        sequence = list()
        # Append a baseline parameter vector to sequence
        sequence.append(deepcopy(baselines))
        # Loop through elements of increments and construct a corresponding entry in sequence
        for key in increments.keys():
            temp = deepcopy(baselines)
            temp[key] = temp[key] + increments[key]
            sequence.append(temp)
        # Return sequence
        return sequence

    def run_sequence(self, parameter_sequence, **kwargs):
        """
        Repeatedly synthesizes copies of the stimulus based on the sequence of parameters provided by
        parameter sequence. kwargs allows the user to pass additional keyword arguments to synthesize()

        Arguments:
            parameter_sequence (list): list of parameter values to synthesize. Each element of the list is a dict of
            parameter names and values that match the expected keyword arguments in synthesize()

        Returns:
            results: list of ndarrays containing synthesized copies of the stimulus
        """
        # Create empty storage for synthesized copies of stimulus
        results = []
        # Loop through the parameter sequence and pass it to synthesize(), unpacking the parameter dict in place
        for parameter_vector in parameter_sequence:
            results.append(self.synthesize(**parameter_vector, **kwargs))
        return results


class Queue:
    """
    Queue handles the logic of how to run multiple simulations in sequence or parallel for a Simulator
    """
    def __init__(self):
        self.queue = []
