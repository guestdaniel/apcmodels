import apcmodels.simulation as si
import apcmodels.synthesis as sy
import apcmodels.anf as anf
import numpy as np


def test_anf_rate_level_function():
    """ Test to make sure that a basic anf simulation can be set up and run on a single sequence of pure tones with
     increasing levels and that the simulation returns a corresponding increasing rate response. Check both the
      Heinz et al. (2001) and the Zilany et al. (2014) nerve model. """
    # Check both AuditoryNerveHeinz2001Numba and AuditoryNerveZilany2014
    for anf_model in [anf.AuditoryNerveHeinz2001Numba, anf.AuditoryNerveZilany2014]:
        # Initialize simulator object
        sim = anf_model()

        # Define stimulus parameters
        fs = int(200e3)  # sampling rate, Hz
        tone_freq = 1000  # tone frequency, Hz
        tone_dur = 0.1  # tone duration, s
        tone_ramp_dur = 0.01  # ramp duration, s
        tone_levels = [0, 10, 20, 30, 40, 50]  # levels to test, dB SPL
        cf_low = 1000  # cf of auditory nerve, Hz
        cf_high = 1000  # cf of auditory nerve, Hz
        n_cf = 1  # how many auditory nerves to test, int

        # Encode parameters in Parameters
        params = si.Parameters(freq=tone_freq, dur=tone_dur, dur_ramp=tone_ramp_dur,
                               fs=fs, cf_low=cf_low, cf_high=cf_high, n_cf=n_cf)
        params.wiggle('level', tone_levels)

        # Add stimuli to Parameters
        params.add_inputs(sy.PureTone().synthesize_sequence(params))

        # Run model
        output = sim.run(params)
        means = [np.mean(resp) for resp in output]  # calculate mean of each response

        assert np.all(np.diff(means) > 0)
