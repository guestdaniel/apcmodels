`apcmodels` is a Python library that implements computational models of the auditory system, simulates responses for these models to a range of stimuli used in psychophysical experiments, and analyzes the outputs of these models. `apcmodels` utilizes code from a range of existing Python packages in the auditory modeling domain and provides a single unified API to access these models. 

# Auditory models

| Model | Locus | Outputs |
| ------ | ------ | ------ |
| [Heinz, Colburn, and Carney (2001)](https://doi.org/10.1162/089976601750541804) | Auditory nerve | Firing rate
| [Zilany, Bruce, and Carney (2014)](https://doi.org/10.1121/1.4837815) | Auditory nerve | Firing rate, spikes
| [Verhulst, Altoe, and Vasilkov (2018)](https://doi.org/10.1016/j.heares.2017.12.018) | Basilar membrane, inner hair cells, auditory nerve | Vibration, potentials, firing rate, spikes


### Heinz, Colburn, and Carney (2001)

The Heinz et al. (2001) model simulates the instantaneous firing rate of HSR auditory nerve fibers. The model as specified in the corresponding paper has been implemented entirely in Python. The auditory nerve stage of the model has been implemented in Numba to improve computation time. 

### Zilany, Bruce, and Carney (2014)

The Zilany et al. (2014) model simulates the instantaneous firing rate of LSR, MSR, and HSR auditory nerve fibers. The model is originally implemented in C, and the [`cochlea`](https://github.com/mrkrd/cochlea) package provides a Python interface to a slightly modified version of that C implementation (the changes are supposed to consist only of removal of `.mex`-specific parts of the C code). We have in turn included a number of files from [`cochlea`](https://github.com/mrkrd/cochlea) package in `external` so that their version of the Zilany et al. (2014) model can be accessed through `apcmodels` without the need to install [`cochlea`](https://github.com/mrkrd/cochlea). The Python interface depends on `Cython` and requires a working C compiler. In theory, installing `apcmodels` should automatically handle the compilation process behind the scenes for you. If this does not work, you may want to try running `python setup.py build_ext --inplace` in the home directory of `apcmodels`, which should run `cythonize()` on the necessary files in `external/zilany2014`. 

### Verhulst, Altoe, and Vasilkov (2018)

The Verhulst et al. (2018) model simulates basilar membrane responses using a transmission line model as well as inner hair cell responses and HSR, MSR, and LSR auditory nerve instantaneous firing rates. 

logic of how to construct batches
- Start with an empty dict
- Use wiggle() to create desired combinations of parameters to test
- Use flatten() to flatten the constructed parameters list 
- Use repeat() to specify how many repetitions you want simulated for each combination of parameters
- Use evaluate() to evaluate any random variables in the parameters
- Use increment() to specify any incremented parameters to test at the bottom level

