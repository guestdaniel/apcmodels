import numpy as np
from gammatone import filters
from scipy.signal import butter, lfilter
from apcmodels.simulation import Simulator
from numba import jit
from cochlea.zilany2014.zilany2014_rate import run_zilany2014_rate


class AuditoryNerveHeinz2001Numba(Simulator):
    def __init__(self):
        super().__init__()

    def simulate(self, params):
        """
        Passes params to the Heinz et al. (2001) firing rate simulation as kwargs and returns the firing rates

        Arguments:
            params: encoded parameters, should be a dict containing parameter names and values

        Returns:
            output (ndarray): output array of instantaneous firing rates, of shape (n_cf, n_sample)
        """
        return calculate_heinz2001_firing_rate(**params)


class AuditoryNerveZilany2014(Simulator):
    def __init__(self):
        super().__init__()

    def simulate(self, params):
        """
        Passes params to the Zilany et al. (2014) firing rate simulation as kwargs and returns the firing rates

        Arguments:
            params: encoded parameters, should be a dict containing parameter names and values

        Returns:
            output (ndarray): output array of instantaneous firing rates, of shape (n_cf, n_sample)
        """
        return calculate_zilany2014_firing_rate(**params)


class AuditoryNerveVerhulst2018(Simulator):
    def __init__(self):
        super().__init__()

    def simulate(self, params):
        """
        Passes params to the Verhulst et al. (2001) firing rate simulation as kwargs and returns the firing rates

        Arguments:
            params: encoded parameters, should be a dict containing parameter names and values

        Returns:
            output (ndarray): output array of instantaneous firing rates, of shape (n_cf, n_sample)
        """
        return calculate_verhulst2018_firing_rate(**params)


def calculate_auditory_nerve_firing_rate(nerve_model):
    """
    A function wrapper around functions that simulate auditory nerve firing rates to handle parameters

    Arguments:
        nerve_model (function): a function that implements a firing rate simulation for an auditory nerve model

    Returns:
        run_model (function): the wrapped function

    """
    def run_model(**kwargs):
        # Handle default inputs
        if 'fs' not in kwargs:
            kwargs['fs'] = int(200e3)  # if no fs, set to default of 200 kHz

        # Append 10 ms silence to the beginning of the acoustic input and the end of acoustic input
        kwargs['_input'] = np.concatenate([np.zeros(int(kwargs['fs']*0.01)),
                                          kwargs['_input'],
                                          np.zeros(int(kwargs['fs']*0.01))])

        # If a user passes 'cfs' and 'cf_low' or 'cf_high', reject the input combination as invalid
        if ('cfs' in kwargs and 'cf_low' in kwargs) or ('cfs' in kwargs and 'cf_high' in kwargs):
            return ValueError('Both `cfs` and `cf_low` or `cf_high` passed at same time')
        # If a user passes cf_low and cf_high along with n_cf, return a log-spaced CF array
        elif 'cf_low' in kwargs and 'cf_high' in kwargs and 'n_cf' in kwargs:
            cfs = 10 ** np.linspace(np.log10(kwargs['cf_low']), np.log10(kwargs['cf_high']), kwargs['n_cf'])
            kwargs['cfs'] = cfs
            kwargs.pop('cf_low')
            kwargs.pop('cf_high')
            kwargs.pop('n_cf')
        # If user passes no appropriate inputs, raise error
        elif 'cfs' not in kwargs:
            raise ValueError('Valid CFs not specified')

        # Pass input to nerve model
        return nerve_model(**kwargs)
    return run_model


@calculate_auditory_nerve_firing_rate
def calculate_heinz2001_firing_rate(_input, fs, cfs=None, **kwargs):
    """
    Implements Heinz, Colburn, and Carney (2001) auditory nerve simulation.

    Arguments:
        _input (ndarray): 1-dimensional ndarray containing an acoustic stimulus in pascals

        fs (int): sampling rate in Hz

        cfs (ndarray): ndarray containing characteristic frequencies at which to simulate responses

    Returns:
        output (ndarray): output array of instantaneous firing rates, of shape (n_cf, n_samp)

    Warnings:
        - Note that arguments passed to **kwargs are discarded silently

    Citations:
        Heinz, M. G., Colburn, H. S., and Carney, L. H. (2001). "Evaluating auditory performance limits: I.
        One-parameter discrimination using a computational model for the auditory nerve." *Neural computation* 13(10).
        2273-2316.
    """
    # Check if cfs is None, if so set 1000 Hz single CF
    if cfs is None:
        cfs = np.array([1000])

    # Run peripheral filters
    bm = filters.erb_filterbank(_input, filters.make_erb_filters(fs, cfs))

    # Saturating nonlinearity
    K = 1225
    beta = -1
    ihc = (np.arctan(K * bm + beta) - np.arctan(beta)) / (np.pi / 2 - np.arctan(beta))

    # Lowpass
    [b, a] = butter(1, 4800 / (fs / 2))
    for ii in range(7):
        ihc = lfilter(b, a, ihc, axis=1)

    # Neural adaptation
    dims = ihc.shape
    C_I = np.zeros_like(ihc)
    C_L = np.zeros_like(ihc)
    return _calculate_heinz2001_rate_internals(dims, fs, ihc, C_I, C_L)


@jit
def _calculate_heinz2001_rate_internals(dims, fs, ihc, C_I, C_L):
    """ Function to implement neural adaptation stage of Heinz (2001) auditory nerve model. Separate from main function
    so that it can be processed with Numba """
    # Neural adaptation
    len_t = dims[1]
    len_f = dims[0]
    T_s = 1 / fs
    r_o = 50
    V_I = 0.0005
    V_L = 0.005
    P_G = 0.03
    P_L = 0.06
    PI_rest = 0.012
    PI_max = 0.6  # not sure why this is unused in Heinz et al. (2001)
    C_G = 6666.7
    P_I = 0.0173 * np.log(1 + np.exp(34.657 * ihc))

    C_I[:, 0] = r_o / PI_rest
    C_L[:, 0] = C_I[:, 0] * (PI_rest + P_L) / P_L

    # Implement dynamics
    for ii in range(len_t - 1):
        for kk in range(len_f):
            C_I[kk, ii + 1] = C_I[kk, ii] + (T_s / V_I) * (
                    -P_I[kk, ii] * C_I[kk, ii] + P_L * (C_L[kk, ii] - C_I[kk, ii]))
            C_L[kk, ii + 1] = C_L[kk, ii] + (T_s / V_L) * (
                    -P_L * (C_L[kk, ii] - C_I[kk, ii]) + P_G * (C_G - C_L[kk, ii]))

    # Return
    return P_I * C_I


@calculate_auditory_nerve_firing_rate
def calculate_zilany2014_firing_rate(_input, fs, cfs=None, species='human', fiber_types='hsr', **kwargs):
    """
    Implements Zilany, Bruce, and Carney (2014) auditory nerve simulation.

    Arguments:
        _input (ndarray): 1-dimensional ndarray containing an acoustic stimulus in pascals

        fs (int): sampling rate in Hz

        cfs (ndarray): ndarray containing characteristic frequencies at which to simulate responses

        fiber_types (list, str): list of fiber types to simulate, or a single fiber type to simulate (from 'hsr', 'msr',
             'lsr'). Requesting multiple types naturally multiplies the number of output channels.

    Returns:
        output (ndarray): output array of instantaneous firing rates, of shape (n_cf, n_samp)

    Warnings:
        - Note that arguments passed to **kwargs are discarded silently

    Citations:
        Zilany, M. S., Bruce, I. C., & Carney, L. H. (2014). Updated parameters and expanded simulation options for a
        model of the auditory periphery. The Journal of the Acoustical Society of America, 135(1), 283-286.
    """
    # Check if cfs is None, if so set 1000 Hz single CF
    if cfs is None:
        cfs = np.array([1000])

    rates = run_zilany2014_rate(_input, fs, anf_types=fiber_types, cf=cfs, cohc=1, cihc=1, species=species,
                                powerlaw='actual', ffGn=False)
    rates = np.array(rates).T
    return rates
