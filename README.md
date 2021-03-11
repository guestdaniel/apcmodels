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
| [Verhulst, Altoe, and Vasilkov (2018)](https://doi.org/10.1016/j.heares.2017.12.018) | Basilar membrane, inner hair cells, auditory nerve | Vibration, potentials, firing rate

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
│   │   └── verhulst2018     # Wrapper around original implementation of Verhulst et al. (2018) AN model
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
├── LICENSE                  # Attributions for source code adapted from other authors (in apcmodels/external)
├── LICENSE                  # License file for the code contained in this repository (excluding code in apcmodels/external)
├── README.md                # This README file
└── setup.py                 # Setup/install file
```

# Installation

Installation of `apcmodels` involves a few simple steps.

1. Download this Git repository. This can be done either with...

```
git clone git@github.com:guestdaniel/apcmodels
```

... or by downloading the [zip file](https://github.com/guestdaniel/apcmodels/archive/master.zip). 

2. [Install `Cython`](https://cython.readthedocs.io/en/latest/src/quickstart/install.html). 

3. Install the [`gammatone`](https://github.com/detly/gammatone) package. This can be done easily with `pip` as...

```
pip install git+https://github.com/detly/gammatone.git
```

4. Install [`Verhulstetal2018Model`](https://github.com/HearingTechnology/Verhulstetal2018Model), the codebase for the Verhulst et al. (2018) nerve model. Note that this repository is not a Python package and, as such, cannot be installed through package managers. Moreover, it requires some manual installation steps. This means it must be located on your computer, set up correctly, and *available on your Python path* to be usable by `apcmodels`. First, clone/download the repository to a preferred location on your device. This can be done from the repository as...

```
git clone git@github.com:HearingTechnology/Verhulstetal2018Model.git 
```

... or by unzipping the [zip file](https://github.com/HearingTechnology/Verhulstetal2018Model/archive/master.zip). Then, navigate to this new local copy of the model and proceed with the second and third step of the installation instructions provided in the [`Verhulstetal2018Model`](https://github.com/HearingTechnology/Verhulstetal2018Model) repository (compile `tridiag.so` and unzip Poles, respectively). Finally, before importing any modules that depend on the Verhulst et al. (2018) code, you must make sure that your Python path includes the directory where you downloaded [`Verhulstetal2018Model`](https://github.com/HearingTechnology/Verhulstetal2018Model). If you are a Linux/MacOS user and run Python through the terminal, something like this in your `.bashrc` or `.tschrc` file will likely work:

```
export PYTHONPATH="${PYTHONPATH}:/my/path/to/Verhulst/model"
```

Alternatively, at the beginning of each session or at the top of every script you run, you may use:

```
import sys
sys.path.append('/my/path/to/Verhulst/model')
```

If you are a PyCharm user, you can separately configure the path of each Python interpreter that you use (see this [StackExchange post](https://stackoverflow.com/questions/48947494/add-directory-to-python-path-in-pycharm)).

Note that importing `apcmodels` should still succeed even if you have not correctly configured the `Verhulstetal2018Model`. However, a warning will be printed to the screen and running any code that depends on the associated modules will fail.

5. Run `python setup.py install` in the root directory of this repository. This will install the package on your computer and automatically install all remaining Python dependencies (e.g., `numpy`, `scipy`) if they are not already installed.

We generally recommend that you install `apcmodels` in a virtual environment (e.g., Conda) to insulate it from your system Python install. 

# Attribution and licensing

`apcmodels` is open-source and would not be possible without the efforts of open-source contributors and scientists around the world. `apcmodels` used modified versions of parts of other open-source projects. Such modified code is stored in `apcmodels/external` (although other code is stored there too!). If a subfoder/submodule in `external` contains modified code from an external source, an appropriate `LICENSE` file is included. More information about this, as well as references to the original authors, can be found in the top-level `AUTHORS.md` file in this repository.

`apcmodels` also depends on the scientific contributions of those who build computational mdoels of the auditory system. Attributions for such scientific works can be found in the doc strings of associated source code.

# Similar tools and resources

Many other resources exist for performing auditory computational modeling. Most of these resources are written in MATLAB, although some are also written in Python. Some examples are listed below.

- [Matlab Auditory Periphery (MAP)](http://www.essexpsychology.webmate.me/HearingLab/modelling.html)
- [Auditory Modeling Toolbox](http://amtoolbox.sourceforge.net/)
- [cochlea](https://github.com/mrkrd/cochlea)

