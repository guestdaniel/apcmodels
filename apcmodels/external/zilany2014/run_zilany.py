from __future__ import division, print_function, absolute_import
import itertools
import numpy as np
import pandas as pd
from apcmodels.external.zilany2014 import _zilany2014
from apcmodels.external.zilany2014.util import calc_cfs


def run_zilany2014_spikes(sound, fs, anf_num, cf, species, seed, cohc=1, cihc=1, powerlaw='approximate', ffGn=False):
    """
    Run the inner ear model of Zilany et al. (2014). This model is based on the original implementation provided by the
    authors. The MEX specific code was replaced by Python code in C files. This variant returns spike trains.

    Parameters:
        sound (ndarray): The input sound in Pa.
        fs (int): Sampling frequency of the sound in Hz.
        anf_num (tuple): The desired number of auditory nerve fibers per frequency channel (CF), (HSR#, MSR#, LSR#).
            For example, (100, 75, 25) means that we want 100 HSR fibers, 75 MSR fibers and 25 LSR fibers per CF.
        cf (ndarray): The center frequency(s) of the simulated auditory nerve fibers. If float, then defines a single
            frequency channel. If array_like (e.g. list or ndarray), then the frequencies are used. If tuple, then must
             have exactly 3 elements (min_cf, max_cf, num_cf) and the frequencies are calculated using the Greenwood
             function.
        species (str): Species of the simulation, either 'cat', 'human', or'human_glasberg1990'
        seed (int): Random seed for the spike generator.
        cohc (float): Degradation of the outer hair cells in [0, 1]
        cihc (float): Degradation of the inner hair cells in [0, 1]
        powerlaw (str) Defines which power law implementation should be used, either 'actual' or 'approximate'
        ffGn (bool): Enable/disable factorial Gaussian noise.

    Returns
        spike_trains (pd.dataframe): Auditory nerve spike trains in a dataframe
    """
    # Check to make sure that inputs are reasonable
    if np.max(sound) > 1000:
        raise ValueError('Signal should be in Pa')
    if sound.ndim != 1:
        raise ValueError('Sound input should be 1d ndarray')
    if species not in ('cat', 'human', 'human_glasberg1990'):
        raise ValueError('Species is not recognized')

    # Set random seed
    np.random.seed(seed)

    # Calculate CFs
    cfs = calc_cfs(cf, species)

    # Set the parameters for each channel
    channel_args = [{'signal': sound, 'cf': cf, 'fs': fs, 'cohc': cohc, 'cihc': cihc, 'anf_num': anf_num,
                     'powerlaw': powerlaw, 'seed': seed, 'species': species, 'ffGn': ffGn} for cf in cfs]

    # Run model for each channel
    nested = map(_run_channel_spikes, channel_args)

    # Unpack the results
    trains = itertools.chain(*nested)
    spike_trains = pd.DataFrame(list(trains))

    return spike_trains


def run_zilany2014_rate(sound, fs, anf_types, cf, species, cohc=1, cihc=1, powerlaw='approximate', ffGn=False):
    """
    Run the inner ear model of Zilany et al. (2014). This model is based on the original implementation provided by the
    authors. The MEX specific code was replaced by Python code in C files. This variant returns firing rates.

    Parameters:
        sound (ndarray): The input sound in Pa.
        fs (int): Sampling frequency of the sound in Hz.
        anf_types (str, list): The desired auditory nerve fiber types, either a list of types or a single type passed
            as a string. Options are 'hsr', 'msr', and 'lsr'
        cf (ndarray): The center frequency(s) of the simulated auditory nerve fibers. If float, then defines a single
            frequency channel. If array_like (e.g. list or ndarray), then the frequencies are used. If tuple, then must
             have exactly 3 elements (min_cf, max_cf, num_cf) and the frequencies are calculated using the Greenwood
             function.
        species (str): Species of the simulation, either 'cat', 'human', or'human_glasberg1990'
        cohc (float): Degradation of the outer hair cells in [0, 1]
        cihc (float): Degradation of the inner hair cells in [0, 1]
        powerlaw (str) Defines which power law implementation should be used, either 'actual' or 'approximate'
        ffGn (bool): Enable/disable factorial Gaussian noise.

    Returns
        spike_trains (pd.dataframe): Auditory nerve spike trains in a dataframe
    """
    # Check to make sure that inputs are reasonable
    if np.max(sound) > 1000:
        raise ValueError('Signal should be in Pa')
    if sound.ndim != 1:
        raise ValueError('Sound input should be 1d ndarray')
    if species not in ('cat', 'human', 'human_glasberg1990'):
        raise ValueError('Species is not recognized')

    # Calculate CFs
    cfs = calc_cfs(cf, species)

    # Check to see if anf_types needs to be placed in a list
    if isinstance(anf_types, str):
        anf_types = [anf_types]

    # Set the parameters for each channel
    channel_args = [{'signal': sound, 'cf': cf, 'fs': fs, 'cohc': cohc, 'cihc': cihc, 'anf_types': anf_types,
                     'powerlaw': powerlaw, 'species': species, 'ffGn': ffGn} for cf in cfs]

    # Run model for each channel
    nested = map(_run_channel_rates, channel_args)

    # Unnest results and prepare for output
    results = list(itertools.chain(*nested))

    columns = pd.MultiIndex.from_tuples(
        [(r['anf_type'],r['cf']) for r in results],
        names=['anf_type','cf']
    )
    rates = np.array([r['rate'] for r in results]).T

    rates = pd.DataFrame(
        rates,
        columns=columns
    )

    return rates


def _run_channel_spikes(args):
    # Unpack parameters
    fs = args['fs']
    cf = args['cf']
    signal = args['signal']
    cohc = args['cohc']
    cihc = args['cihc']
    powerlaw = args['powerlaw']
    seed = args['seed']
    anf_num = args['anf_num']
    species = args['species']
    ffGn = args['ffGn']

    # Run BM, IHC
    vihc = _zilany2014.run_ihc(
        signal=signal,
        cf=cf,
        fs=fs,
        species=species,
        cohc=float(cohc),
        cihc=float(cihc)
    )

    # Calculate duration
    duration = len(vihc) / fs
    anf_types = np.repeat(['hsr', 'msr', 'lsr'], anf_num)

    # Create empty output objects
    synout = {}
    trains = []

    # Loop through anf types
    for anf_type in anf_types:

        if (anf_type not in synout) or ffGn:
            # Run synapse
            synout[anf_type] = _zilany2014.run_synapse(
                fs=fs,
                vihc=vihc,
                cf=cf,
                anf_type=anf_type,
                powerlaw=powerlaw,
                ffGn=ffGn
            )

        # Run spike generator
        spikes = _zilany2014.run_spike_generator(
            synout=synout[anf_type],
            fs=fs,
        )

        # Add spike train to output
        trains.append({
            'spikes': spikes,
            'duration': duration,
            'cf': args['cf'],
            'type': anf_type
        })

    return trains


def _run_channel_rates(args):
    # Unpack parameters
    fs = args['fs']
    cf = args['cf']
    signal = args['signal']
    cohc = args['cohc']
    cihc = args['cihc']
    powerlaw = args['powerlaw']
    anf_types = args['anf_types']
    species = args['species']
    ffGn = args['ffGn']


    # Run BM, IHC
    vihc = _zilany2014.run_ihc(
        signal=signal,
        cf=cf,
        fs=fs,
        species=species,
        cohc=float(cohc),
        cihc=float(cihc)
    )

    # Create empty output object
    rates = []

    # Loop through ANF types
    for anf_type in anf_types:
        # Run synapse
        synout = _zilany2014.run_synapse(
            fs=fs,
            vihc=vihc,
            cf=cf,
            anf_type=anf_type,
            powerlaw=powerlaw,
            ffGn=ffGn
        )

        # Append rates to results
        rates.append({
            'rate': synout / (1 + 0.75e-3*synout),
            'cf': cf,
            'anf_type': anf_type
        })

    return rates
