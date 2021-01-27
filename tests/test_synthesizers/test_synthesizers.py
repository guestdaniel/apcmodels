from apcmodels import synthesizers as sy
import apcmodels.signal as sg
import numpy as np


def test_puretone_synthesize():
    """ Check that PureTone object can successfully synthesize and replicates standard puretone synthesis"""
    synth = sy.PureTone()
    output = synth.synthesize(1000, 50, 0, 1, 0.1, int(48e3))
    reference = sg.cosine_ramp(sg.scale_dbspl(sg.pure_tone(1000, 0, 1, int(48e3)), 50), 0.1, int(48e3))
    assert np.all(output == reference)


def test_puretone_run_sequence():
    """ Check that puretone's run_sequence() correctly accepts a list of dicts and returns a corresponding number of
    stimuli """
    synth = sy.PureTone()
    # Generate 1000 and 2000 Hz pure tones using run_sequence
    results = synth.run_sequence([{'freq': 1000, 'level': 50, 'phase': 0, 'dur': 1, 'fs': int(48e3), 'dur_ramp': 0.1},
                                  {'freq': 2000, 'level': 50, 'phase': 0, 'dur': 1, 'fs': int(48e3), 'dur_ramp': 0.1}])
    # Generate references manually
    reference1 = sg.cosine_ramp(sg.scale_dbspl(sg.pure_tone(1000, 0, 1, int(48e3)), 50), 0.1, int(48e3))
    reference2 = sg.cosine_ramp(sg.scale_dbspl(sg.pure_tone(2000, 0, 1, int(48e3)), 50), 0.1, int(48e3))
    assert np.all(results[0] == reference1) and np.all(results[1] == reference2)


def test_puretone_incremented_sequence():
    """ Check that puretone's run_sequence() and create_incremented_sequence() successfully combine to produce two
    pure tones, one with a slightly higher frequency"""
    synth = sy.PureTone()
    # Generate 1000 and 2000 Hz pure tones using run_sequence
    results = synth.run_sequence(synth.create_incremented_sequence(baselines={'freq': 1000},
                                                                   increments={'freq': 0.001}))
    # Generate references manually
    reference1 = sg.cosine_ramp(sg.scale_dbspl(sg.pure_tone(1000, 0, 1, int(48e3)), 50), 0.1, int(48e3))
    reference2 = sg.cosine_ramp(sg.scale_dbspl(sg.pure_tone(1000.001, 0, 1, int(48e3)), 50), 0.1, int(48e3))
    assert np.all(results[0] == reference1) and np.all(results[1] == reference2)


