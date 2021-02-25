"""
The following script replicates the main figure results found in Heinz et al. (2001) (Figure 4). This serves to validate
the present implementation of the Heinz et al. (2001) auditory nerve model and the ideal observer calculation framework.

The key results figure (Figure 4) estimates FDLs as a function of frequency (panel a), duration (panel b), and
level (panel c) for pure tones. The simulations were conducted at 500 kHz. More information about the simulations used
in each panel of the figure is available below.

Heinz, M. G., Colburn, H. S., and Carney, L. H. (2001). "Evaluating auditory performance limits: I. One-parameter
discrimination using a computational model for the auditory nerve." *Neural computation* 13(10). 2273-2316.
"""
import apcmodels.synthesis as sy
import apcmodels.simulation as si
import apcmodels.signal as sg
import apcmodels.anf as anf
import apcmodels.decode as dc
import numpy as np
import matplotlib.pyplot as plt


class PureToneHeinz2001(sy.Synthesizer):
    """
    Synthesizes a pure tone with raised-cosine ramps according to the specifications of Heinz et al. (2001)
    """
    def __init__(self):
        super().__init__(stimulus_name='Pure Tone Heinz (2001)')

    def synthesize(self, freq=1000, level=50, phase=0, dur=1, dur_ramp=0.1, fs=int(500e3), **kwargs):
        """
        Synthesizes a single instance of a scaled copy of a pure tone with a raised-cosine ramp. Stimulus duration is
        defined as the duration between the half-amplitude points of the ramps.

        Arguments:
            freq (float): frequency of pure tone in Hz
            level (float): level of pure tone in dB SPL
            phase (float): phase offset in degrees, must be between 0 and 360
            dur (float): duration in seconds
            dur_ramp (float): duration of raised-cosine ramp in seconds
            fs (int): sampling rate in Hz

        Returns:
            output (array): pure tone
        """
        pt = sg.pure_tone(freq, phase, dur+dur_ramp, fs)
        pt = sg.scale_dbspl(pt, level)
        pt = sg.cosine_ramp(pt, dur_ramp, fs)
        return pt


"""
Panel A

FDLs in terms of Weber fraction (delta_f/f) as a function of frequency for 40 dB SPL pure tones with 200 ms
duration and 20 ms rise/fall times (240 ms duration in total).

The results from apcmodels closely matches the results from Heinz et al. (2001). Two notable exceptions are visible. 
First, the all-information thresholds are marginally higher in our simulations than in Heinz et al. (2001)... slight
differences in the duration or effective level of presentation may account for this? Second, the rate-place thresholds
exhibit some minor "jitter", which I believe can be chalked up to the fact that we do not match pure-tone frequencies
to CFs (as was done in Heinz et al.). Thus the effective "spatial undersampling" by only sampling some CFs can distort
the estimated thresholds, and this distortion may be worse for some frequencies than others. 
"""


def calculate_fdl_vs_frequency(f_low, f_high, n_f):
    """
    Calculates ideal observer FDL vs frequency

    Arguments:
        f_low (float): lowest frequency to test in Hz

        f_high (float): highest frequency to test in Hz

        n_f (int): number of frequencies to test... frequencies between f_low and f_high will be distributed
            logarithmically

    Returns:
        tuple of ndarrays, the first containing all-information thresholds, the second containing
            rate-place thresholds, and the third containing the frequencies at which they were estimated

    """
    # Initialize simulator object
    sim = anf.AuditoryNerveHeinz2001Numba()

    # Define stimulus parameters
    tone_level = 40
    tone_dur = 0.20
    tone_ramp_dur = 0.02
    tone_freqs = 10**np.linspace(np.log10(f_low), np.log10(f_high), n_f)

    # Encode stimulus parameters
    params = {'level': tone_level, 'dur': tone_dur, 'dur_ramp': tone_ramp_dur}
    params = si.wiggle_parameters(params, 'freq', tone_freqs)

    # Encode model parameters
    params = si.append_parameters(params, ['cf_low', 'cf_high', 'n_cf', 'fs', 'n_fiber_per_chan', 'delta_theta', 'API'],
                                  [100, 10000, 60, int(500e3), 200, [0.001], np.zeros(1)])

    # Encode increment and synthesize
    params = si.increment_parameters(params, {'freq': 0.001})
    synth = PureToneHeinz2001()
    stimuli = synth.synthesize_sequence(params)
    params = si.stitch_parameters(params, '_input', stimuli)

    # Run model
    output = sim.run(params,
                     parallel=True,
                     runfunc=dc.decode_ideal_observer(sim.simulate))
    t_AI = [x[0] for x in output]  # extract AI thresholds, which are always the first element of each tuple in results
    t_RP = [x[1] for x in output]  # extract RP thresholds, which are always the second element of each tuple in results

    # Return
    return np.array(t_AI), np.array(t_RP), tone_freqs


# Calculate figure 4a
t_ai, t_rp, f = calculate_fdl_vs_frequency(200, 10000, 20)
# Make figure
plt.figure(figsize=(3.5, 6))
# Plot AI results
plt.scatter(f/1000, t_ai/f, marker='s', color='black')
plt.plot(f/1000, t_ai/f, linestyle='dashed', color='black')
# Plot RP results
plt.scatter(f/1000, t_rp/f, marker='o', color='black')
plt.plot(f/1000, t_rp/f, linestyle='dashed', color='black')
# Handle scales and limits
plt.yscale('log')
plt.xscale('log')
plt.ylim((1e-6, 1e-2))
# Add labels
plt.xlabel('Frequency (kHz)')
plt.ylabel('Weber Fraction')
plt.title('Panel a')


"""
Panel B

FDLs in terms of difference limen in Hz as a function of duration for 40 dB SPL pure tones at 970 Hz and with 4 ms
rise/fall times. 
 
The results from apcmodels closely matches the results from Heinz et al. (2001). Two notable exceptions are visible. 
First, the all-information thresholds are marginally higher in our simulations than in Heinz et al. (2001)... slight
differences in the duration or effective level of presentation may account for this? Second, the rate-place simulations
have a slightly different curve (more concave than convex) than the Heinz et al. simulations, but the differences are 
fairly minor.
"""


def calculate_fdl_vs_dur(dur_low, dur_high, n_dur):
    """
    Calculates ideal observer FDL vs duration

    Arguments:
        dur_low (float): lowest duration to test in seconds

        dur_high (float): highest duration to test in seconds

        n_dur (int): number of durations to test... frequencies between dur_low and dur_high will be distributed
            logarithmically

    Returns:
        tuple of ndarrays, the first containing all-information thresholds, the second containing
            rate-place thresholds, and the third containing the durations at which they were estimated

    """
    # Initialize simulator object
    sim = anf.AuditoryNerveHeinz2001Numba()

    # Define stimulus parameters
    tone_level = 40
    tone_freq = 970
    tone_ramp_dur = 0.004
    tone_durs = 10**np.linspace(np.log10(dur_low), np.log10(dur_high), n_dur)

    # Encode stimulus parameters
    params = {'level': tone_level, 'freq': tone_freq, 'dur_ramp': tone_ramp_dur}
    params = si.wiggle_parameters(params, 'dur', tone_durs)

    # Encode model parameters
    params = si.append_parameters(params, ['cf_low', 'cf_high', 'n_cf', 'fs', 'n_fiber_per_chan', 'delta_theta', 'API'],
                                  [100, 10000, 60, int(500e3), 200, [0.001], np.zeros(1)])

    # Encode increment and synthesize
    params = si.increment_parameters(params, {'freq': 0.001})
    synth = PureToneHeinz2001()
    stimuli = synth.synthesize_sequence(params)
    params = si.stitch_parameters(params, '_input', stimuli)

    # Run model
    output = sim.run(params,
                     parallel=True,
                     runfunc=dc.decode_ideal_observer(sim.simulate))
    t_AI = [x[0] for x in output]  # extract AI thresholds, which are always the first element of each tuple in results
    t_RP = [x[1] for x in output]  # extract RP thresholds, which are always the second element of each tuple in results

    # Return
    return np.array(t_AI), np.array(t_RP), tone_durs


# Calculate figure 4b
t_ai, t_rp, d = calculate_fdl_vs_dur(0.004, 0.50, 8)
# Make figure
plt.figure(figsize=(3.5, 6))
# Plot AI results
plt.scatter(d*1000, t_ai, marker='s', color='black')
plt.plot(d*1000, t_ai, linestyle='dashed', color='black')
# Plot RP results
plt.scatter(d*1000, t_rp, marker='o', color='black')
plt.plot(d*1000, t_rp, linestyle='dashed', color='black')
# Handle scales and limits
plt.yscale('log')
plt.xscale('log')
plt.ylim((5e-4, 2e1))
# Add labels
plt.xlabel('Duration (ms)')
plt.ylabel('Difference Limen (Hz)')
plt.title('Panel b')


"""
Panel C

FDLs in terms of difference limen in Hz as a function of level for pure tones at 970 Hz with 2 ms rise/fall ramps with
durations of 500 ms. In Heinz et al. (2001) the simulations were conducted in terms of SL, but for simplicity here we 
conducted the simulations in dB SPL. Thus, this figure does not perfectly capture the results in Heinz et al. but 
demonstrates that conceptually increases in level produce the same effective change in the response. 
"""


def calculate_fdl_vs_level(level_low, level_high, n_level):
    """
    Calculates ideal observer FDL vs level

    Arguments:
        level_low (float): lowest level to test in dB SPL

        level_high (float): highest level to test in dB SPL

        n_level (int): number of levels to test... frequencies between level_low and level_high will be distributed
            linearly

    Returns:
        tuple of ndarrays, the first containing all-information thresholds, the second containing
            rate-place thresholds, and the third containing the levels at which they were estimated

    """
    # Initialize simulator object
    sim = anf.AuditoryNerveHeinz2001Numba()

    # Define stimulus parameters
    tone_dur = 0.50
    tone_freq = 970
    tone_ramp_dur = 0.002
    tone_levels = np.linspace(level_low, level_high, n_level)

    # Encode stimulus parameters
    params = {'dur': tone_dur, 'freq': tone_freq, 'dur_ramp': tone_ramp_dur}
    params = si.wiggle_parameters(params, 'level', tone_levels)

    # Encode model parameters
    params = si.append_parameters(params, ['cf_low', 'cf_high', 'n_cf', 'fs', 'n_fiber_per_chan', 'delta_theta', 'API'],
                                  [100, 10000, 60, int(500e3), 200, [0.001], np.zeros(1)])

    # Encode increment and synthesize
    params = si.increment_parameters(params, {'freq': 0.001})
    synth = PureToneHeinz2001()
    stimuli = synth.synthesize_sequence(params)
    params = si.stitch_parameters(params, '_input', stimuli)

    # Run model
    output = sim.run(params,
                     parallel=True,
                     runfunc=dc.decode_ideal_observer(sim.simulate))
    t_AI = [x[0] for x in output]  # extract AI thresholds, which are always the first element of each tuple in results
    t_RP = [x[1] for x in output]  # extract RP thresholds, which are always the second element of each tuple in results

    # Return
    return np.array(t_AI), np.array(t_RP), tone_levels


# Calculate figure 4b
t_ai, t_rp, l = calculate_fdl_vs_level(0, 60, 8)
# Make figure
plt.figure(figsize=(3.5, 6))
# Plot AI results
plt.scatter(l, t_ai, marker='s', color='black')
plt.plot(l, t_ai, linestyle='dashed', color='black')
# Plot RP results
plt.scatter(l, t_rp, marker='o', color='black')
plt.plot(l, t_rp, linestyle='dashed', color='black')
# Handle scales and limits
plt.yscale('log')
plt.ylim((5e-4, 2e1))
plt.xlim((-5, 85))
# Add labels
plt.xlabel('Level (dB SPL)')
plt.ylabel('Difference Limen (Hz)')
plt.title('Panel c')
