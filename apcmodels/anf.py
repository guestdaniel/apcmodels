import numpy as np
from gammatone import filters
from scipy.signal import butter, lfilter
from apcmodels.simulation import Simulator, check_args
from numba import jit
from apcmodels.external.zilany2014.run_zilany import run_zilany2014_rate, run_zilany2014_spikes
from apcmodels.external.verhulst2018.run_verhulst import run_verhulst2018_rate
import warnings


class AuditoryNerveHeinz2001Numba(Simulator):
    """ Provides an interface to the Heinz, Colburn, and Carney (2001) auditory nerve model.

    References:
        Heinz, M. G., Colburn, H. S., and Carney, L. H. (2001). "Evaluating auditory performance limits: I.
        One-parameter discrimination using a computational model for the auditory nerve." Neural Computation 13(10).
        2273-2316.
    """
    def __init__(self):
        super().__init__()
        # Declare recognized parameters for this model
        self.known_params = ['_input', 'fs', 'cfs', 'cf_low', 'cf_high', 'n_cf']

    @check_args([])
    def simulate(self, params, **kwargs):
        """ Runs the Heinz et al. (2001) auditory nerve simulation and return firing rates

        Args:
            params (dict): encoded parameters, should be a dict containing parameter names and values
            kwargs: keyword arguments not encoded in params, passed through to firing rate calculation function below

        Returns:
            output (ndarray): output array of instantaneous firing rates, of shape (n_cf, n_sample)

        Notes:
            - check_args() decorator above raises warnings if any keys in params/kwargs are not in the signature of
                calculate_heinz2001_firing_rate. If you have warnings enabled, you may see these warnings. This is
                useful for debugging in cases where you are not sure if certain parameters are being accepted/used by
                downstream functions.
        """
        return calculate_heinz2001_firing_rate(**params, **kwargs)


class AuditoryNerveZilany2014(Simulator):
    """ Provides an interface to the Zilany, Bruce and Carney (2014) auditory nerve model.

    The Zilany et al. (2014) model is implemented via code adapted from the cochlea package
    (https://github.com/mrkrd/cochlea)

    References:
        Zilany, M. S., Bruce, I. C., & Carney, L. H. (2014). Updated parameters and expanded simulation options for a
        model of the auditory periphery. The Journal of the Acoustical Society of America, 135(1), 283-286.
    """
    def __init__(self):
        super().__init__()
        # Declare recognized parameters
        self.known_params = ['_input', 'fs', 'cfs', 'cf_low', 'cf_high', 'n_cf']

    @check_args(['species', 'fiber_type'])
    def simulate(self, params, **kwargs):
        """ Runs the Zilany et al. (2014) auditory nerve simulation and return firing rates

        Args:
            params (dict): encoded parameters, should be a dict containing parameter names and values
            kwargs: keyword arguments not encoded in params, passed through to firing rate calculation function below

        Returns:
            output (ndarray): output array of instantaneous firing rates, of shape (n_cf, n_sample)

        Notes:
            - check_args() decorator above raises warnings if any keys in params/kwargs are not in the signature of
                calculate_heinz2001_firing_rate. If you have warnings enabled, you may see these warnings. This is
                useful for debugging in cases where you are not sure if certain parameters are being accepted/used by
                downstream functions.
        """
        return calculate_zilany2014_firing_rate(**params, **kwargs)

    @check_args(['species', 'anf_num'])
    def simulate_spikes(self, params, **kwargs):
        """ Runs the Zilany et al. (2014) auditory nerve simulation and return spikes

        Args:
            params (dict): encoded parameters, should be a dict containing parameter names and values
            kwargs: keyword arguments not encoded in params, passed through to spikes calculation function below

        Returns:
            output (pd.dataframe): output dataframe of spike times, of shape (n_cf*n_fiber, ...)

        Notes:
            - check_args() decorator above raises warnings if any keys in params/kwargs are not in the signature of
                calculate_heinz2001_firing_rate. If you have warnings enabled, you may see these warnings. This is
                useful for debugging in cases where you are not sure if certain parameters are being accepted/used by
                downstream functions.
        """
        return calculate_zilany2014_spikes(**params, **kwargs)


class AuditoryNerveVerhulst2018(Simulator):
    """ Provides an interface to the Verhulst, Altoe, and Vasilikov (2018) auditory nerve model

    The Verhulst et al. (2018) model is implemented via a tiny wrapper around the Verhulst lab's release of their 2018
    model available at https://github.com/HearingTechnology/Verhulstetal2018Model

    References:
        Verhulst, S., Altoè, A., & Vasilkov, V. (2018). Computational modeling of the human auditory periphery:
        Auditory-nerve responses, evoked potentials and hearing loss. Hearing research, 360, 55-75.
    """
    def __init__(self):
        super().__init__()
        # Declare recognized parameters
        self.known_params = ['_input', 'fs', 'cfs', 'cf_low', 'cf_high', 'n_cf']

    @check_args([])
    def simulate(self, params, **kwargs):
        """ Runs the Verhulst et al. (2018) auditory nerve simulation and return firing rates

        Args:
            params (dict): encoded parameters, should be a dict containing parameter names and values

        Returns:
            output (ndarray): output array of instantaneous firing rates, of shape (n_cf, n_sample)
        """
        return calculate_verhulst2018_firing_rate(**params, **kwargs)


def calculate_auditory_nerve_response(nerve_model):
    """ A wrapper around functions that simulate auditory nerve responses to handle parameters

    Examines the kwargs passed to nerve_model() and handles a number of potential issues/case scenarios in the inputs:
        - If cfs is not a kwarg, then cfs is constructed from cf_low, cf_high, and n_cf
        - If cfs is a kwarg, but any of cf_low, cf_high, and n_cf are also kwargs, an error is raised
        - A default sampling rate of 200 kHz is set if none is provided (a bit dangerous to rely on this)
        - 5 ms of silence is appended to the beginning of the stimulus
        - 40 ms of silence is appended to the end of the stimulus

    Args:
        nerve_model (function): a function that implements an auditory nerve model simulation, accepting various kwargs
            and returning anything

    Returns:
        run_model (function): the wrapped function

    """
    def run_model(**kwargs):
        # Handle default inputs
        if 'fs' not in kwargs:
            kwargs['fs'] = int(200e3)  # if no fs, set to default of 200 kHz

        # Append 5 ms silence to the beginning of the acoustic input and 40 ms to the end of acoustic input
        kwargs['_input'] = np.concatenate([np.zeros(int(kwargs['fs']*0.005)),
                                          kwargs['_input'],
                                          np.zeros(int(kwargs['fs']*0.040))])

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


@calculate_auditory_nerve_response
def calculate_heinz2001_firing_rate(_input, fs, cfs, **kwargs):
    """ Runs the Heinz et al. (2001) auditory nerve simulation and return firing rates

    Implements the Heinz et al. (2001) auditory nerve model. This model contains the following steps:
        - A gammatone frontend is implemented via the gammatone package (https://github.com/detly/gammatone)
        - A saturating nonlinearity simulating the actions of the inner hair cells (IHC) is applied
        - The IHC responses are lowpass filtered with 7 first-order Butterworth filters
        - Auditory nerve responses to the IHC inputs are simulated - this stage is implemented via Numba for speed.
            The implementation described in Heinz et al. (2001) is a slightly simplified version of three-stage
            diffusion as in Westerman and Smith (1988).

    Most of the parameter descriptions below in the inline documentation are taken directly from Heinz et al. (2001).

    Args:
        _input (ndarray): 1-dimensional ndarray containing an acoustic stimulus in pascals
        fs (int): sampling rate in Hz
        cfs (ndarray): ndarray containing characteristic frequencies at which to simulate responses

    Returns:
        output (ndarray): output array of instantaneous firing rates, of shape (n_cf, n_samp)

    Warnings:
        - Arguments passed via **kwargs are silently unused

    References:
        Heinz, M. G., Colburn, H. S., and Carney, L. H. (2001). "Evaluating auditory performance limits: I.
        One-parameter discrimination using a computational model for the auditory nerve." Neural Computation, 13(10).
        2273-2316.
        Westerman, L. A., & Smith, R. L. (1988). A diffusion model of the transient response of the cochlear inner hair
        cell synapse. The Journal of the Acoustical Society of America, 83(6), 2266-2276.
    """
    # Calculate peripheral filter outputs
    bm = filters.erb_filterbank(_input, filters.make_erb_filters(fs, cfs))

    # Apply saturating nonlinearity
    K = 1225   # controls sensitivity
    beta = -1  # sets 3:1 asymmetric bias
    ihc = (np.arctan(K * bm + beta) - np.arctan(beta)) / (np.pi / 2 - np.arctan(beta))

    # Apply lowpass filter
    [b, a] = butter(1, 4800 / (fs / 2))
    for ii in range(7):
        ihc = lfilter(b, a, ihc, axis=1)

    # Apply auditory nerve + neural adaptation stage
    dims = ihc.shape
    C_I = np.zeros_like(ihc)  # immediate concentration ("spikes/volume")
    C_L = np.zeros_like(ihc)  # local concentration ("spikes/volume")
    return _calculate_heinz2001_rate_internals(dims, fs, ihc, C_I, C_L)


@jit
def _calculate_heinz2001_rate_internals(dims, fs, ihc, C_I, C_L):
    """ Implements the auditory nerve fiber stage adaptation stage of Heinz (2001) auditory nerve model

    This function simulated auditory nerve fibers as in Heinz et al. (2001), and is separate from main function that
    implementes the basilar membrane and inner hair cell frontend to permit fast implementation via Numba's jit()
    decorator.

    Args:
        dims (tuple): shape of the IHC input
        fs (int): sampling rate in Hz
        ihc (ndarray): array of inner hair cells responses, of shape (n_chan, n_sample)
        C_I (ndarray): empty array of same shape as ihc
        C_L (ndarray): empty array of same sahpe as ihc

    Returns:
        output (ndarray): output array of same size as ihc, instantaneous firing rate of high spontaneous rate auditory
            nerve fibers at each time sample
    """
    # Neural adaptation
    len_t = dims[1]
    len_f = dims[0]
    T_s = 1 / fs     # sampling period
    r_o = 50         # spontaneous discharge rate
    V_I = 0.0005     # immediate "volume"
    V_L = 0.005      # local "volume"
    P_G = 0.03       # global permeability ("volume"/s)
    P_L = 0.06       # local permeability ("volume"/s)
    PI_rest = 0.012  # resting immediate permeability ("volume"/s)
    PI_max = 0.6     # maximum immediate permeability ("volume"/s")... not sure why this is unused in paper eq.
    C_G = 6666.7     # global concentration ("spikes/volume")
    P_I = 0.0173 * np.log(1 + np.exp(34.657 * ihc))  # immediate permeability ("volume"/s)

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


@calculate_auditory_nerve_response
def calculate_zilany2014_firing_rate(_input, fs, cfs=None, species='human', fiber_type='hsr', **kwargs):
    """ Runs the Zilany et al. (20144) auditory nerve simulation and returns firing rates

    The Zilany et al. (2014) model is implemented via code adapted from the cochlea package
    (https://github.com/mrkrd/cochlea).

    Args:
        _input (ndarray): 1-dimensional ndarray containing an acoustic stimulus in pascals
        fs (int): sampling rate in Hz
        cfs (ndarray): ndarray containing characteristic frequencies at which to simulate responses. Note that the
            Zilany model for humans cannot support cfs that are below 125 Hz or above 20 kHz. If a user passes CFs in
            that range, they are clipped to within [125, 20000] and a warning is raised accordingly.
        species (str): species of simulation, either 'cat' or 'human'
        fiber_type (list, str): list of fiber types to simulate, or a single fiber type to simulate (from 'hsr', 'msr',
             'lsr'). Requesting multiple types naturally multiplies the number of output channels.

    Returns:
        output (ndarray): output array of instantaneous firing rates, of shape (n_cf, n_sample)

    Warnings:
        - Note that arguments passed via **kwargs are silently unused

    References:
        Zilany, M. S., Bruce, I. C., & Carney, L. H. (2014). Updated parameters and expanded simulation options for a
        model of the auditory periphery. The Journal of the Acoustical Society of America, 135(1), 283-286.
    """
    # Set CFs that are too high or too low to min/max values
    if np.min(cfs) < 125:
        cfs[cfs < 125] = 125
        warnings.warn('CFs below 125 Hz were set to 125 Hz!!')
    if np.max(cfs) > 20000:
        cfs[cfs > 20000] = 20000
        warnings.warn('CFs above 20 kHz were set to 20 kHz!!')

    # Run firing rate simulation using cochlea package
    rates = run_zilany2014_rate(_input, fs, anf_types=fiber_type, cf=cfs, cohc=1, cihc=1, species=species,
                                powerlaw='approximate', ffGn=False)  # note power law and ffGn are fast versions
    rates = np.array(rates).T  # transpose to (n_cf, n_sample)
    return rates


@calculate_auditory_nerve_response
def calculate_zilany2014_spikes(_input, fs, cfs=None, species='human', anf_num=(1, 0, 0), **kwargs):
    """ Runs the Zilany et al. (20144) auditory nerve simulation and returns spike times

    The Zilany et al. (2014) model is implemented via code adapted from the cochlea package
    (https://github.com/mrkrd/cochlea).

    Args:
        _input (ndarray): 1-dimensional ndarray containing an acoustic stimulus in pascals
        fs (int): sampling rate in Hz
        cfs (ndarray): ndarray containing characteristic frequencies at which to simulate responses
        species (str): species of simulation, either cat or human
        anf_num (tuple): tuple indicating how many of each fiber type to simulate ('hsr', 'msr', 'lsr')

    Returns:
        output (pd.dataframe): output dataframe of spike times, of shape (n_cf*n_fiber, ...)

    Warnings:
        - Note that arguments passed to **kwargs are discarded silently

    References:
        Zilany, M. S., Bruce, I. C., & Carney, L. H. (2014). Updated parameters and expanded simulation options for a
        model of the auditory periphery. The Journal of the Acoustical Society of America, 135(1), 283-286.
    """
    # Set CFs that are too high or too low to min/max values
    if np.min(cfs) < 125:
        cfs[cfs < 125] = 125
        warnings.warn('CFs below 125 Hz were set to 125 Hz!!')
    if np.max(cfs) > 20000:
        cfs[cfs > 20000] = 20000
        warnings.warn('CFs above 20 kHz were set to 20 kHz!!')

    # Run firing rate simulation using cochlea package
    spikes = run_zilany2014_spikes(_input, fs, anf_num=anf_num, cf=cfs, cohc=1, cihc=1, species=species,
                                   powerlaw='approximate', ffGn=False)  # note power law and ffGn are fast versions
    return spikes


@calculate_auditory_nerve_response
def calculate_verhulst2018_firing_rate(_input, fs, cfs=None, **kwargs):
    """ Implements Verhulst, Altoe, and Vasilikov (2018) auditory nerve simulation

    The Verhulst et al. (2018) model is implemented via a tiny wrapper around the Verhulst lab's release of their 2018
    model available at https://github.com/HearingTechnology/Verhulstetal2018Model

    Args:
        _input (ndarray): 1-dimensional ndarray containing an acoustic stimulus in pascals
        fs (int): sampling rate in Hz
        cfs (ndarray): ndarray containing characteristic frequencies at which to simulate responses. Note that the
            Verhulst model returns responses at a hardcoded range of 1000 CFs.. For each requested CF, we return
            the closest available CF. This can produce significant distortion along the tonotopic axis or can even
            result in the same response from a single CF being returned multiple times, so consider what CFs you
            request with caution.

    Returns:
        output (ndarray): output array of instantaneous firing rates, of shape (n_cf, n_sample)

    Warnings:
        - Note that arguments passed to **kwargs are discarded silently
        - The CFs provided do not necessarily align exactly with the true underlying CFs of the model. Care should be
            taken for any application that depends crucially on the assumed underlying CFs. In such applications,
            you may want to make measurements of the CFs yourself.

    References:
        Verhulst, S., Altoè, A., & Vasilkov, V. (2018). Computational modeling of the human auditory periphery:
        Auditory-nerve responses, evoked potentials and hearing loss. Hearing research, 360, 55-75.
    """
    # Run firing rate simulation
    rates, cfs_model = run_verhulst2018_rate(_input, fs, 2)
    rates = np.flip(rates, axis=1)  # flip tonotopic axis left-right (by default it goes high->low)
    rates = rates.T  # transpose to (n_cf, n_sample)

    # Only return rates at requested "CFs" (while not forgetting to flip model CFs)
    idxs = np.array([np.argmin(np.abs(np.flip(cfs_model) - cf_requested)) for cf_requested in cfs])
    return rates[idxs, :]
