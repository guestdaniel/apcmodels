# apcmodels

`apcmodels` is a Python 3 library developed by Daniel Guest in the [Auditory Perception and Cognition Lab](http://apc.psych.umn.edu/) at the University of Minnesota. `apcmodels` implements computational models of the auditory system, simulates responses for these models to acoustic stimuli, and analyzes the outputs of these models. `apcmodels` utilizes code from a range of existing Python packages/code in the auditory modeling domain and provides a single unified API to access these models. Moreover, `apcmodels` features extensive unit testing and thorough documentation to ensure validity of its outputs and ease of use.

# Features

- Access multiple auditory models with a unified interface
- Rapidly generate batch simulations over ranges of parameters or combinations of parameters
- Automatically utilize parallel processing for batch simulations
- Decode responses with ideal observers ([Heinz, Colburn, and Carney, 2001](https://doi.org/10.1162/089976601750541804))
- Ensure reliability and validity of results with extensive testing

# Implemented models

| Model | Locus | Outputs |
| ------ | ------ | ------ |
| [Heinz, Colburn, and Carney (2001)](https://doi.org/10.1162/089976601750541804) | Auditory nerve | Firing rate
| [Zilany, Bruce, and Carney (2014)](https://doi.org/10.1121/1.4837815) | Auditory nerve | Firing rate, spikes
| [Verhulst, Altoe, and Vasilkov (2018)](https://doi.org/10.1016/j.heares.2017.12.018) | Basilar membrane, inner hair cells, auditory nerve | Vibration, potentials, firing rate, spikes

# Upcoming models

| Model | Locus | Outputs |
| ------ | ------ | ------ |
| [Krips and Furst (2009)](https://www.mitpressjournals.org/doi/full/10.1162/neco.2009.07-07-563) | "Midbrain"-type neurons | Firing rate

# Examples

### Rate-level function

Tools provided in `apcmodels.simulation` create a simple and readable interface for setting up simulations. Here, we set up, run, and plot a simulation of a simple rate-level function for a single HSR auditory nerve fiber tuned to 1000 Hz. The simulation by default is parallelized across multiple cores. The `params` object encodes the exact stimulus and model parameters used in each element of `output` which allows users to easily perform *post hoc* analyses and document the details of their simulations.

```python
import apcmodels.simulation as si
import apcmodels.synthesis as sy
import apcmodels.anf as anf
import numpy as np
import matplotlib.pyplot as plt

# Set up simulation
sim = anf.AuditoryNerveHeinz2001Numba()

# Define stimulus parameters
fs = int(200e3)                         # sampling rate, Hz
tone_freq = 1000                        # tone frequency, Hz
tone_dur = 0.1                          # tone duration, s
tone_ramp_dur = 0.01                    # ramp duration, s
tone_levels = [0, 10, 20, 30, 40, 50]   # levels to test, dB SPL
cf_low = 1000                           # cf of auditory nerve, Hz
cf_high = 1000                          # cf of auditory nerve, Hz
n_cf = 1                                # how many auditory nerves to test, int

# Encode parameters in Parameters
params = si.Parameters(freq=tone_freq, dur=tone_dur, dur_ramp=tone_ramp_dur,
                       fs=fs, cf_low=cf_low, cf_high=cf_high, n_cf=n_cf)
params.wiggle('level', tone_levels)

# Add stimuli to Parameters
params.add_inputs(sy.PureTone().synthesize_sequence(params))

# Run model
output = sim.run(params)
means = [np.mean(resp) for resp in output]  # calculate mean of each response

# Plot
plt.plot(tone_levels, means)
plt.xlabel('Level (dB SPL)')
plt.ylabel('Response (sp/s)')
```

# File structure

`apcmodels` is organized as one main package (`apcmodels`) and a handful of subpackages that implement functionality adapted from external libraries/packages (`external`). Tests are located in three subfolders; unit tests are in `test_unit`, acceptance tests are in `test_acceptance`, and replications of figures or results from the literature are in `test_replication`. 

```
.  
├── apcmodels                # Primary package directory
│   ├── external             # Submodule that contains code from external sources (e.g., other packages)
│   │   ├── zilany2014       # Code adapted from cochlea package to implement Zilany et al. (2014) AN model
│   │   └── verhulst2018     # Code adapted from CoNNear package to implement Verhulst et al. (2018) AN model
│   ├── __init__.py          
│   ├── anf.py               # Auditory nerve models 
│   ├── decode.py            # Ideal observer analysis 
│   ├── signal.py            # Signal processing essentials and some psychophysics stimuli
│   ├── simulation.py        # Core simulation interface 
│   ├── synthesis.py         # Core acoustic stimulus generation interface
│   ├── util.py              # Miscellaneous functions and utilities
│   └── ...      
├── test_acceptance          # Acceptance tests
├── test_unit                # Unit tests
├── test_replication         # Replications from literature
├── LICENSE                  # License file for the code contained in this repository
├── README.md                # This README file
└── setup.py                 # Setup/install file
```

# Installation

To install `apcmodels`, simply download this Git repository and run `python setup.py install` in the root directory. Alternatively, you can simply install the package directly from git using `pip` as...
```
pip install git+https://github.com/guestdaniel/apcmodels.git
```
We generally recommend that you install `apcmodels` in a virtual environment (e.g., Conda) to insulate it from your system Python install. A number of Python packages are required but should be installed automatically by the install script. One notable exception is the [`gammatone`](https://github.com/detly/gammatone) package, which must be manually installed or installed via `pip` as...

```
pip install git+https://github.com/detly/gammatone.git
```

# Similar tools and resources

Many other resources exist for performing auditory computational modeling. Most of these resources are written in MATLAB, although some are also written in Python. Some examples are listed below.

- [Matlab Auditory Periphery (MAP)](http://www.essexpsychology.webmate.me/HearingLab/modelling.html)
- [Auditory Modeling Toolbox](http://amtoolbox.sourceforge.net/)
- [cochlea](https://github.com/mrkrd/cochlea)

