import apcmodels.signal as sg
import numpy as np
from apcmodels.simulation import Parameters


class Synthesizer:
    """ Provides an interface for defining a synthesis routine for an acoustic stimulus and running it at scale.

    Args:
        stimulus_name (string): human-readable name of the stimulus type
    """
    def __init__(self, stimulus_name='none'):
        self.stimulus_name = stimulus_name

    def synthesize(self, **kwargs):
        """ Method to synthesize the stimulus at set parameter values

        Expected to be overridden by subclasses and contain the essential synthesis code to implement a particular
        acoustic stimulus.
        """
        return kwargs

    def _synthesize_preprocess_inputs(self, parameter_vector, **kwargs):
        """ Handles any arguments in a dict of parameters before passing it onto synthesize()

        Method to use before synthesize(), handles any input parameters that are callables and not numbers or strings.
        If users pass functions as parameter values, this function ensures that they are evaluated before synthesis.
        A major use case for this is to allow the user to pass random variables instead of fixed parameter values. Then,
         on each iteration of a simulation, a new random value will be generated.

        Args:
            parameter_vector (dict): dict of parameter names and values. Some values may be callable, if so they are
                evaluated now. Values that are not callable are passed onward without being altered.
            kwargs: keyword arguments not encoded in parameter_vector. These are passed onto synthesize() without
                being altered.

        Returns:
            output: output of synthesize()
        """
        # Loop through elements of the parameter vector and call them if they're callable
        for key in parameter_vector:
            if callable(parameter_vector[key]):
                parameter_vector[key] = parameter_vector[key]()
        # Call synthesize
        return self.synthesize(**parameter_vector, **kwargs)

    def synthesize_sequence(self, parameters, **kwargs):
        """ Repeatedly synthesizes copies of the stimulus based on a sequence of parameters

        parameters is expected to encode a series of acoustic stimuli (each as a dict of parameter names and values).
        synthesize_sequence() iterates through these encodings, synthesizes the corresponding stimulus, and returns
        all the stimuli in the same shape as the input parameters sequence.

        Args:
            parameters (dict, list, ndarray): dict of parameter names and values, or a list or ndarray of such
                dicts. If the element is a list, synthesize_sequence() is applied to each element. If an ndarray,
                synthesize_sequence() is applied to each element.
            kwargs: keyword arguments not encoded in parameters, passed through without being touched to synthesize()

        Returns:
            results (list, ndarray): list or array containing synthesized copies of the stimulus. The list/array
                structure will mimic that of the input parameter sequence. Each dict in parameter_sequence becomes a
                single ndarray in results.
        """
        # If we have a Parameters object, extract params
        if type(parameters) is Parameters:
            parameters = parameters.params
        # If parameters is an array, loop through it using nditer
        if type(parameters) is np.ndarray:
            temp = np.empty(parameters.shape, dtype=object)
            with np.nditer(parameters, flags=['refs_ok', 'multi_index'], op_flags=['readwrite']) as it:
                for x in it:
                    temp[it.multi_index] = self.synthesize_sequence(x.item(), **kwargs)
            return temp
        # If parameters is a list, process with list comprehension recursively
        elif type(parameters) is list:
            return [self.synthesize_sequence(params, **kwargs) for params in parameters]
        # If parameters is a dict, synthesize the stimulus
        elif type(parameters) is dict:
            return self._synthesize_preprocess_inputs(parameters, **kwargs)
        else:
            raise TypeError('parameter_sequence should be an array, a list, or a dict')


class PureTone(Synthesizer):
    """ Synthesizes a pure tone. """
    def __init__(self):
        super().__init__(stimulus_name='Pure Tone')

    def synthesize(self, freq=1000, level=50, phase=0, dur=1, dur_ramp=0.1, fs=int(48e3), **kwargs):
        """ Synthesizes a single instance of a scaled copy of a pure tone with a raised-cosine ramp.

        Args:
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


class ComplexTone(Synthesizer):
    """ Synthesizes a simple complex tone stimulus with a consecutive series of harmonics. """
    def __init__(self):
        super().__init__(stimulus_name='Complex Tone')

    def synthesize(self, f0=100, h_low=1, h_high=10, level=50, phase=0, dur=1, dur_ramp=0.1, fs=int(48e3), **kwargs):
        """ Synthesizes a single instance of a complex tone with a raised-cosine ramp.

        The complex tone is synthesized with a consecutive series of harmonics from h_low to h_high. All components
        have the same level and the same phase.

        Args:
            f0 (float): F0 of complex tone, in Hz
            h_low (int): lowest harmonic number
            h_high (int): highest harmonic number
            level (float): level of complex tone per-component, in dB SPL
            phase (float): phase offset for every component, in degrees
            dur (float): duration, in seconds
            dur_ramp (float): duration of raised-cosine ramp, in seconds
            fs (int): sampling rate, in Hz

        Returns:
            output (array): complex tone
        """
        freqs = np.arange(h_low, h_high+1) * f0
        levels = np.ones(len(freqs))*level
        phases = np.zeros(len(freqs))
        pt = sg.complex_tone(freqs, levels, phases, dur, fs)
        pt = sg.cosine_ramp(pt, dur_ramp, fs)
        return pt
