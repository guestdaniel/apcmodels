import numpy as np
from gammatone import filters
from scipy.signal import butter, lfilter


def calculate_auditory_nerve_firing_rate(nerve_model):
    def run_model(**kwargs):
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
def calculate_Heinz2001_firing_rate(input, cfs=None, fs=int(200e3)):
    # Check if cfs is None, if so set 1000 Hz single CF
    if cfs is None:
        cfs = np.array([1000])

    # Run peripheral filters
    bm = filters.erb_filterbank(input, filters.make_erb_filters(fs, cfs))

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
    len_t = dims[1]
    len_f = dims[0]
    T_s = 1 / fs
    r_o = 50
    V_I = 0.0005
    V_L = 0.005
    P_G = 0.03
    P_L = 0.06
    PI_rest = 0.012
    PI_max = 0.6
    C_G = 6666.7
    P_I = 0.0173 * np.log(1 + np.exp(34.657 * ihc))
    C_I = np.zeros_like(ihc)
    C_L = np.zeros_like(ihc)
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
