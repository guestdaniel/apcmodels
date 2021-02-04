def decode_ideal_observer(ratefunc):
    """
    Estimate thresholds for calculating an ideal observer

    Args:
        ratefunc (function): a function that accepts **kwargs and returns firing rates for a neural simulation.

    Returns:

    """
    def inner(params):
        """
        Runs ratefunc on each input encoded in params, then estimates thresholds based on an ideal observer for a
        particular parameter. This requires some additional information to be encoded in params in the form of an
        a priori information matrix (API) and

        Args:
            params (dict): parameters and inputs encoded in a dict. In the minimal case-scenario, params will contain
                just an input element with a single input ndarray and some model parameters. In more complex scenarios,
                the input element may contain multiple ndarrays. In either case, we use ratefunc to simulate firing
                rates for all inputs in params and then apply ideal observer analysis to decode accordingly.

        Returns:
            thresholds (ndarray): predicted all-information and rate-place thresholds
        """
        rates = run_rates_util(params, ratefunc)

        return DOSOMETHING(rates)
    return inner


def run_rates_util(ratefunc, _input, **kwargs):
    """
    Takes a list of inputs and passes each element to ratefunc along with kwargs, returns the results as a list

    Arguments:
        ratefunc (function): a function that accepts input and other kwargs and returns model simulations

        _input: an input or a list of inputs

    Returns:
        output: results of applying ratefunc to each input in params

    """
    # If the input is not a list, just run ratefunc
    if type(_input) is not list:
        return [ratefunc(_input=_input, **kwargs)]
    # If the input *is* a list, process each input separately
    else:
        output = []
        for _input_element in _input:
            output.append(ratefunc(_input=_input_element, **kwargs))
    return output
