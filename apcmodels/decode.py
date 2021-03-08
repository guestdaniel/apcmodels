import numpy as np
from copy import deepcopy


def decode_ideal_observer(ratefunc):
    """ Decode response by estimating thresholds for an ideal observer.

    Estimates thresholds for a particular parameter in a set of parameter based on simulated responses to those
    parameters. Implemented as a wrapper that can be applied to any ratefunc that accepts a dict of parameter names
    and values (params) and returns a single firing rate simulation in the standard style.

    The basic approach is as follows. We assume that, instead of encoding a single simulation as a dict of parameter
    names and values (as any Simulator's simulate() method or ratefunc would assume), multiple simulations have been
    encoded as a (possibly nested) list of such dicts. We refer to this as params (see below). The neural rate response
    is simulated for each encoded set of parameters using ratefunc. Then, ideal observer thresholds are derived from
    these simulated rate responses.

    --- possible nesting types

    --- strategies

    --- assumptions

    Args:
        ratefunc (function): a function that accepts **kwargs and returns firing rates for a neural simulation

    Returns:
        inner (function): the wrapped ratefunc
    """
    def inner(params):
        """ Runs ratefunc on each input encoded in params and estimates ideal observer thresholds

        inner() runs ratefunc on each input/parameter set encoded in params. Then, it calculates ideal observer
        thresholds from those simulations. This requires some additional information to be encoded in params above and
        beyond a standard neural simulation. Namely, we need
            - API (ndarray): An a priori information matrix (API)
            - delta_theta (ndarray, list): a collection of values indicating how much various parameters were
                incremented
            - n_fiber_per_chan (ndarray, list): a collection of values indicating how many fibers are "represented" at
                by each CF's rate response

        Args:
            params (list, ndarray): a collection of dicts of parameter names and values. See docstring for the function
                above to see how this should be structured.

        Returns:
            thresholds (tuple): predicted all-information and rate-place thresholds
        """
        # Pull parameters from encoded list/dict of parameters
        fs = find_parameter(params, 'fs')
        delta_theta = find_parameter(params, 'delta_theta')
        n_fiber_per_chan = find_parameter(params, 'n_fiber_per_chan')
        API = find_parameter(params, 'API')

        # Run ratefunc on kwargs and get firing rates for each input
        rates = run_rates_util(ratefunc, params)

        # Check to see if the elements of rates are ndarrays or lists... if they are not lists, we need to put
        # rates inside a list so it can be processed by the list comprehension below
        if type(rates[0]) is not list:
            rates = [rates]

        # Compute partial derivative matrices for rates for AI and then RP
        pdms_AI = [compute_partial_derivative_matrix(x, fs, delta_theta, n_fiber_per_chan, 'AI') for x in rates]
        pdms_AI = np.array(pdms_AI)

        pdms_RP = [compute_partial_derivative_matrix(x, fs, delta_theta, n_fiber_per_chan, 'RP') for x in rates]
        pdms_RP = np.array(pdms_RP)

        # Return ideal observer results
        return calculate_threshold(pdms_AI, API), calculate_threshold(pdms_RP, API)

    def compute_partial_derivative_matrix(x, fs, delta_theta, n_fiber_per_chan, _type):
        """ Computes a partial derivative matrix as in Siebert (1972) / Heinz et al. (2001)

        Args:
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
        if n_param < 1:
            raise ValueError('There is only one simulation per condition --- ideal observer needs n_param + 1 '
                             'simulations!')
        # Transform from list to ndarray
        x = np.array(x)
        x = np.transpose(x, [1, 0, 2])  # shape: n_cf x (n_param + 1) x n_sample
        # Add small baseline firing rate to avoid issues with zeros and NaNs
        x += 1
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
            deriv_matrix = np.sum(np.transpose(n_fiber_per_chan * np.transpose(deriv_matrix, [1, 2, 0]), [2, 0, 1]), axis=0)
            return deriv_matrix
        elif _type == 'RP':
            # Calculate the duration of the response
            t_max = baseline.shape[2] * 1/fs
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
        """ Optimally decode firing-rate waveforms using N-D generalization of Siebert (1972).

        Args:
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


def run_rates_util(ratefunc, params):
    """
    Takes inputs and processes each element recursively.

    Arguments:
        ratefunc (function): a function that accepts input and other kwargs and returns model simulations

        params (dict, list): inputs and parameters encoded as a dict or list of dicts. If the input is just a single
            dict, we unpack it and pass it directly to ratefunc. Otherwise, we operate recursively on it.

    Returns:
        output: results of applying ratefunc to each input in params

    """
    # If the input is not a list, just run ratefunc
    output = []
    if type(params) is dict:
        return ratefunc(params)
    # If the input *is* a list, process each input separately
    elif type(params) is list:
        for _input_element in params:
            output.append(run_rates_util(ratefunc, _input_element))
    else:
        raise ValueError('params ought to be a dict or a list')
    return output


def find_parameter(params, param_name):
    """
    Searches through params to locate any instance of param_name and returns its value

    Arguments:
        params (dict, list): dict or (possibly nested) list of dicts. Each dict contains parameter names and values.

        param_name (str): parameter name to search for

    Returns:
        value: value of the param_name in the first dict it could be found in
    """
    value = None
    if type(params) is dict:
        if param_name in params.keys():
            return params[param_name]
        else:
            return None
    else:
        for element in params:
            value = find_parameter(element, param_name)
            if value is not None:
                return value


def decode_staircase_procedure(ratefunc, rule, synthesizer, tracked_parameter, starting_value, step_sizes, n_down=2,
                               n_up=1, n_interval=2, transform=lambda x, params: x, max_trials=500, max_num_reversal=12,
                               calculate_from_reversal=6, max_value=1e5, max_num_trials_at_max_value=10):
    """ Estimate a threshold given a rule applied in an adaptive staircase procedure.

    Estimates thresholds for a particular parameter based on a rule applied in an adaptive staircase procedure. 
    Implemented as a wrapper that can be applied to any ratefunc that accepts a dict of parameter names
    and values (params) and returns a single firing rate simulation in the standard style.
    
    The staircase procedure used in a n-up-m-down adaptive procedure with a k-forced-choice decision. On each simulated
    trial, synthesizer() is used to synthesize k copies of the stimulus. The first k-1 copies are "reference" stimuli
    and are synthesized as synthesizer(**params). The final copy is a "target" stimulus and is synthesized as
    synthesizer(**modified_params), where modified_params[tracked_parameter] has been incremented by
    transform(tracked_value, params). In other words, the final copy of the stimulus is identical to the previous k-1
    copies modulo random variation and the tracked parameter (e.g., frequency) being increased according to the current
    value of the tracked variable (e.g., interval size). Then, ratefunc is run on each stimulus to produce a simulated
    neural response. These responses are bundled into a tuple and passed along with params to rule(), which is a
    function that should return an integer value indicating which interval is thought to be the target. The target is
    always the last interval. The staircase procedure continues until one stopping criterion is met (either a maximum
    number of trials, a maximum number of reversals, or a maximum number of trials at the maximum permitted value of
    the tracked variable).

    Args:
        ratefunc (function): a function that accepts params encoded in a dictionary and returns firing rates for a
            neural simulation. Note that we assume that ratefunc takes a single input encoded via
            the '_input' key in the dictionary. However, the user need not (and should not) include a value for
            '_input'. Instead, '_input' values are generated many times throughout the staircase procedure by the
            synthesizer passed to this function.
        rule (function): a function that operates on a tuple of simulations outputs generated by ratefunc as well as
            a params dict (see below) to generate a single integer number indicating which element of the tuple contains
            the "target" stimulus in a n-alternative forced-choice task. The signature should be rule(resp, params), 
            where resp is the tuple of simulations and params is the dict of parameters.
        synthesizer (Synthesizer): A Synthesizer object with a synthesize() method that will accept an unpacked dict of
            parameter names and values and return a single acoustic stimulus. 
        tracked_parameter (str): the name of the parameter associated with the tracked variable in the adaptive
            procedure. For example, if we are tracking 10*log10(delta_f/f), we would pass 'freq'.
        starting_value (float): the starting value of the tracked variable
        step_sizes (list, ndarray): a list or array of step sizes to use in the staircase procedure. The first step
            size is used until the first reversal, at which point the second step size is used and so on. If there are
            fewer step sizes than reversals, the last step size is used until the end of the procedure.
        n_down (int): how many trials need to be correct in sequence before we decrease the size of the tracked variable
        n_up (int): how many trials need to be correct in sequence before we increase the size of the tracked variable
        n_interval (int): how many intervals in each trial
        transform (function): a function that accepts the tracked variable and a dict of parameter names and values with
            the signature transform(var, params) and returns a transformed version of the variable. This is useful when
            the tracked variable is not additive with the associated parameter. Defaults to an identity function.
        max_trials (int): maximum number of trials to test before terminating the procedure
        max_num_reversal (int): maximum number of reversals before terminating the procedure
        calculate_from_reversal (int): how many reversals to use in calculating the threshold
        max_value (float): the maximum value to permit the tracked variable to go to
        max_num_trials_at_max_value (int): how many trials the tracked variable can stay at the max_value before we
            terminate the procedure early.
    """
    def inner(params):
        """
        Wrapper around ratefunc passed above that performs staircase procedure

        Args:
            params (dict): parameters and inputs encoded in a dict or in a list of dicts

        Returns:
            threshold (float): threshold value estimated from staircase procedure
        """
        # Start staircase procedure
        n_trial = 0                     # track number of trials in staircase
        n_trial_at_max = 0              # track number of trials in sequence at the maximum value of tracked variable
        correct = []                    # track correct/incorrect on each trial
        reversals = []                  # track indices of reversals
        direction = []                  # track direction changes (going up or going down)
        values = []                     # track value on each trial
        tracked_value = starting_value  # track the value of the tracking variable
        # Begin while loop
        while n_trial < max_trials and \
                len(reversals) < max_num_reversal and \
                n_trial_at_max < max_num_trials_at_max_value:
            # Add 1 to n_trial
            n_trial += 1
            # Add 1 to n_trial_at_max if needed and clip tracked_value to max_value
            if tracked_value >= max_value:
                tracked_value = max_value
                n_trial_at_max += 1
            else:
                n_trial_at_max = 0
            # Add value to values
            values.append(tracked_value)
            # Test this trial and append result to correct
            correct.append(one_step(params, tracked_value))
            # Handle logic of whether to change value or not
            if correct[-n_up:] == [0]*n_up:
                tracked_value = tracked_value + index_into_step_sizes(len(reversals))  # increase value
                direction.append(1)  # indicate we're going upward
            elif correct[-n_down:] == [1]*n_down:
                tracked_value = tracked_value - index_into_step_sizes(len(reversals))  # decrease value
                direction.append(-1)  # indicate we're going downward
            else:
                if n_trial > 1:
                    direction.append(direction[-1])  # if nothing's happening, we're staying on the same direction
                else:
                    direction.append(0)
            # Handle logic of reversals
            try:
                if np.abs(direction[-2] - direction[-1]) > 1:
                    reversals.append(n_trial-1)  # if directions differ, indicate reversal
            except IndexError:
                continue
        # Once we exit the staircase, return the threshold
        return np.mean([values[ii] for ii in reversals[-calculate_from_reversal:]])

    def generate_stimuli(params, value):
        # Synthesize initial reference stimulus using params
        _input = [synthesizer.synthesize(**params) for rep in range(n_interval-1)]
        # Synthesize initial target stimulus using copy of params
        params_copy = deepcopy(params)
        params_copy[tracked_parameter] = params_copy[tracked_parameter] + transform(value, params)
        _input_target = synthesizer.synthesize(**params_copy)
        # Return
        return *_input, _input_target

    def one_step(params, value):
        # Synthesize initial stimuli
        stim = generate_stimuli(params, value)
        # Synthesize response
        resp = [ratefunc(params, _input=x) for x in stim]
        return rule(resp, params) == len(resp)  # the target is always the last stimulus, so response should match target

    def index_into_step_sizes(idx):
        if idx < len(step_sizes):
            return step_sizes[idx]
        else:
            return step_sizes[-1]

    return inner
