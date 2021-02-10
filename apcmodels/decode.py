import numpy as np


def decode_ideal_observer(ratefunc):
    """
    Estimate thresholds for calculating an ideal observer. Implemented as a wrapper that can be applied to any ratefunc
    that accepts kwargs and returns a single firing rate simulation in the standard apcmodels style.

    Arguments:
        ratefunc (function): a function that accepts **kwargs and returns firing rates for a neural simulation
    """
    def inner(**kwargs):
        """
        Runs ratefunc on each input encoded in params, then estimates thresholds based on an ideal observer for a
        particular parameter. This requires some additional information to be encoded in params in the form of an
        a priori information matrix (API) and

        Arguments:
            params (dict): parameters and inputs encoded in a dict. In the minimal case-scenario, params will contain
                just an input element with a single input ndarray and some model parameters. In more complex scenarios,
                the input element may contain multiple ndarrays. In either case, we use ratefunc to simulate firing
                rates for all inputs in params and then apply ideal observer analysis to decode accordingly.

        Returns:
            thresholds (ndarray): predicted all-information and rate-place thresholds
        """
        # Run ratefunc on kwargs
        rates = run_rates_util(ratefunc, **kwargs)

        # Compute partial derivative matrix for rates
        pdm = compute_partial_derivative_matrix(rates, kwargs['fs'], kwargs['delta_theta'],
                                                kwargs['n_fiber_per_chan'], 'AI')

        # Return ideal observer results
        return calculate_threshold(pdm, kwargs['API'])

    def compute_partial_derivative_matrix(x, fs, delta_theta, n_fiber_per_chan, _type):
        """
        Given one list of simulations, computes a partial derivative matrix as in Siebert (1972).

        Arguments:
            x (list): list of ndarrays containing firing-rate simulations in shape (n_channel x n_sample). The first
                array should be a firing-rate simulation for baseline parameter values. The following arrays should
                be firing-rate simulations where a single parameter has been incremented by a small amount.

            fs (int): sampling rate in Hz

            delta_theta (ndarray): 1d ndarray containing the increment size for each element of x after the first
            n_fiber_per_chan (array): array containing integers of len n_cf, each element indicates how many fibers
                are theoretically represented by the single corresponding channel in x

            _type (str): either 'AI' or 'RP' for all-information or rate-place

        Returns:

        """
        # Calculate n_param
        n_param = len(x)-1
        assert n_param > 0
        # Transform from list to ndarray
        x = np.array(x)
        x = np.transpose(x, [1, 0, 2])  # shape: n_time x (n_param + 1) x n_sample
        # Construct one ndarray of baseline values and another of incremented values
        baseline = np.tile(x[:, 0, :], [n_param, 1, 1])
        baseline = np.transpose(baseline, [1, 0, 2])  # shape: n_cf x n_param x n_sample
        incremented = x[:, 1:, :]  # shape: n_cf x n_param x n_sample
        if _type == 'AI':
            # Estimate derivative with respect to each parameter
            deriv_estimate = np.transpose(np.transpose((incremented - baseline), [0, 2, 1]) / delta_theta, [0, 2, 1])  # shape: n_CF x n_param x n_time
            # Normalize the derivatives by the square root of rate
            deriv_norm = np.sqrt(1 / baseline) * deriv_estimate  # shape: n_CF x n_param x n_time
            # Compute derivative matrix
            deriv_matrix = 1 / fs * np.matmul(deriv_norm, np.transpose(deriv_norm, [0, 2, 1]))  # shape: n_CF x n_param x n_param
            # Sum across fibers
            deriv_matrix = np.sum(np.transpose(n_fiber_per_chan * np.transpose(deriv_matrix, [1, 2, 0]), [2, 0, 1]))
            return deriv_matrix
        elif _type == 'RP':
            # Calculate the duration of the response
            t_max = baseline.shape[0] * 1/fs
            # Average results across time
            baseline = np.mean(baseline, axis=2)
            incremented = np.mean(incremented, axis=2)
            # Estimate derivative with respect to each parameter
            deriv_estimate = (incremented - baseline)/delta_theta
            # Normalize the derivatives by the square root of rate
            deriv_norm = np.sqrt(1 / baseline) * deriv_estimate  # shape: n_CF x n_param
            # Compute derivative matrix
            deriv_norm = np.stack((deriv_norm, deriv_norm), axis=2)
            deriv_matrix = np.matmul(deriv_norm, np.transpose(deriv_norm, [0, 2, 1]))  # shape: n_CF x n_param x n_param
            # Sum across fibers
            deriv_matrix = 0.5 * t_max * np.sum(n_fiber_per_chan * deriv_matrix, axis=0)  # shape: n_param x n_param
            return deriv_matrix

    def calculate_threshold(FI, API):
        """
        Optimally decode firing-rate waveforms of ANFs using N-D generalization from Siebert (1972).

        Arguments:
            FI (ndarray): Fisher information matrices for parameters, of size n_param x n_param

            API (ndarray): Fisher information for parameter distributions, of size n_param x n_param.

        Returns:
            threshold (float): threshold estimate
        """
        if FI.shape == ():
            return np.sqrt(1 / FI)
        elif np.ndim(FI) == 2:
            J = FI + API
            return np.sqrt(np.linalg.inv(J)[0, 0])
        elif np.ndim(FI) == 3:
            FI = np.mean(FI, axis=0)
            J = FI + API
            return np.sqrt(np.linalg.inv(J)[0, 0])

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
