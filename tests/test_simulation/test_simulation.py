import apcmodels.simulation as sy


def test_simulation_run():
    """ Check that Simulation object can successfully run"""
    sim = sy.Simulator()
    sim.run()


def test_synthesize_run_sequence():
    """ Check that run_sequence() correctly accepts a list of dicts and returns a corresponding number of stimuli"""
    synth = sy.Synthesizer()
    results = synth.run_sequence([{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}])
    assert len(results) == 2

def test_synthesize_run_sequence_with_kwarg():
    """ Check that run_sequence() correctly accepts a list of dicts and returns a corresponding number of stimuli while
    also allowing the user to pass additional keyword arguments"""
    synth = sy.Synthesizer()
    results = synth.run_sequence([{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}], qux='hello world')
    assert len(results) == 2

def test_synthesize_run_sequence_with_duplicated_kwarg():
    """ Check that run_sequence() correctly accepts a list of dicts and returns a corresponding number of stimuli, but
    if the user passes a kwarg that is already passed once by the parameter sequence then an error is returned"""
    synth = sy.Synthesizer()
    try:
        synth.run_sequence([{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}], foo='hello world')
        raise Exception
    except TypeError:
        return

def test_synthesizer_synthesize():
    """ Check that Synthesizer object can successfully synthesize"""
    synth = sy.Synthesizer()
    synth.synthesize()