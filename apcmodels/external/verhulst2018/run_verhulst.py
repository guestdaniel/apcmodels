import numpy as np
try:
    import inner_hair_cell2018 as ihc
    import auditory_nerve2018 as anf
    from cochlear_model2018 import cochlea_model
except ModuleNotFoundError:
    print('Warning: You seem not to have the Verhulst et al. (2018) model on your path!')
import pkgutil

# Load in the default poles from Verhulst et al. (2018) model
default_poles = np.array([float(x) for x in
                          pkgutil.get_data('apcmodels.external.verhulst2018', 'default_poles.dat').decode(
                              'utf-8').splitlines()])


def run_verhulst2018_cochlea(stimulus, fs, poles=None):
    """ Calculates basilar membrane stage of Verhuslt et al. (2018) model given acoustic stimulus input

    Args:
        stimulus (ndarray): sound pressure waveform in pascals, of shape (n_sample, )
        fs (int): sampling rate, in Hz
        poles (float, ndarray): either a float or an ndarray of size (n_channel, ). Passed to sheraPo in the Verhulst
            et al. (2018) model. See the source code for more information.

    Returns:
        Vbm (array): basilar membrane vibration, of shape (n_sample, 1000)
        cf (array): nominal characteristic frequencies of cochlear channels, of shape (n_channel, ), in Hz
    """
    # Handle poles
    if poles is None:
        poles = default_poles
    # Initialize and solve cochlear model
    coch = cochlea_model()
    coch.init_model(stimulus, fs, int(1e3), 'all', Zweig_irregularities=np.ones(1),
                    sheraPo=poles, subject=1, IrrPct=0.05, non_linearity_type='vel')
    coch.solve()

    # Return
    return coch.Vsolution, coch.cf


def run_verhulst2018_ihc(bm, fs):
    """ Calculates output of inner hair cell stage of Verhulst et al. (2018) model given basilar membrane input

    Args:
        bm (ndarray): array of basilar membrane velocity (?) values output from run_verhulst2018_cochlea(), of shape
            (n_sample, n_channel)
        fs (int): sampling rate, in Hz

    Returns:
        potential (array): inner hair cell potentials, of shape (n_sample, n_channel)
    """
    potential = ihc.inner_hair_cell_potential(bm * 0.118, fs)
    return potential


def run_verhulst2018_anf(ihc, fs, spont):
    """ Calculates output of the auditory nerve stage of Verhulst et al. (2018) model given inner hair cell input

    Args:
        ihc (ndarray): array of inner hair cell potentials output from run_verhulst2018_ihc(), of shape
            (n_sample, n_channel)
        fs (int): sampling rate, in Hz
        spont (int): which type of auditory nerve fiber to simulate (HSR, MSR, or LSR). Each fiber type is indicated
            by an integer value (0=HSR, 1=MSR, 2=HSR). Values outside of [0, 2] result in an HSR fiber being simulated.

    Returns:
        firing_rates (array): auditory nerve instantaneous firing rates, of shape (n_sample, n_channel)
    """
    return anf.auditory_nerve_fiber(ihc, fs, spont)/(1/fs)  # divide by 1/fs to turn into firing rates


def run_verhulst2018_rate(stimulus, fs, spont):
    """ Calculates output of auditory periphery of the Verhulst et al. (2018) model given acoustic stimulus input

    Args:
        stimulus (ndarray): sound pressure waveform in pascals, of shape (n_sample, )
        fs (int): sampling rate, in Hz
        spont (int): which type of auditory nerve fiber to simulate (HSR, MSR, or LSR). Each fiber type is indicated
            by an integer value (0=HSR, 1=MSR, 2=HSR). Values outside of [0, 2] result in an HSR fiber being simulated.

    Returns:
        anf (array): auditory nerve instantaneous firing rates, of shape (n_sample, 1000)
        cfs (array): nominal characteristic frequencies of each channel, of shape (1000, )
    """
    bm, cfs = run_verhulst2018_cochlea(stimulus, fs)
    ihc = run_verhulst2018_ihc(bm, fs)
    anf = run_verhulst2018_anf(ihc, fs, spont)
    return anf, cfs
