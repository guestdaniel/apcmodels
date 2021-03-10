`apcmodels` is based in part on code adapted from other authors. Here, attributions for these authors are listed along with information about what changes were made from their original code and the licenses that apply to their contributions.

## Marek Rudnicki, Oliver Schoppe, Michael Isik, Florian Völk, and Werner Hemmert (`cochlea`)

**Description**: The present implementation of the Zilany, Bruce, and Carney (2014) auditory nerve model is based on modified copies of source code originally written by Marek Rudnicki and colleagues that is available in the [`cochlea`](https://github.com/mrkrd/cochlea) package. 

**Changes**: A subset of the authors' `cochlea/cochlea/zilany2014` submodule was modified and included in `apcmodels` in March 2020. `.c` and `.pyx` files were largely untouched, while code from `.py` files was refactored into `run_zilany.py`.

**Location**: `apcmodels/external/zilany2014`.

**License**:  `cochlea` is licensed under [GNU GPLv3](https://github.com/mrkrd/cochlea/blob/master/COPYING.txt) and all code in the present package derived from `cochlea` is likewise licensed under GNU GPLv3. A copy of the license is available as `apcmodels/external/LICENSE`. 