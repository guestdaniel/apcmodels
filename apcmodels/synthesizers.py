import apcmodels.signal as sg
from apcmodels.simulation import Synthesizer

class PureTone(Synthesizer):
    """
    Synthesizes pure tones
    """
    def __init__(self):
        super().__init__(stimulus_name='Pure Tone')

    def synthesize(self, freq=1000, level=50, phase=0, dur=1, dur_ramp=0.1, fs=int(48e3)):
        """
        Synthesizes a single instance of a scaled copy of a pure tone with a raised-cosine ramp

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
        pt = sg.pure_tone(freq, phase, dur, fs)
        pt = sg.scale_dbspl(pt, level)
        pt = sg.cosine_ramp(pt, dur_ramp, fs)
        return pt