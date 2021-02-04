from copy import deepcopy
from apcmodels.simulation import run_rates_util


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
