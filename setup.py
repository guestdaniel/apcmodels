from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "apcmodels.external.zilany2014._zilany2014",
        [
            "apcmodels/external/zilany2014/_zilany2014.pyx",
            "apcmodels/external/zilany2014/model_IHC.c",
            "apcmodels/external/zilany2014/model_Synapse.c",
            "apcmodels/external/zilany2014/complex.c"
        ]
    ),
]

setup(
    name='apcmodels',
    version='0.1.0',
    packages=['apcmodels',
              'apcmodels.external',
              'apcmodels.external.zilany2014',
              'apcmodels.external.verhulst2018'],
    license='GPLv3',
    author='Daniel R. Guest',
    install_requires=['numpy', 'scipy', 'pandas', 'matplotlib', 'Cython', 'numba', 'pathos', 'tqdm'],
    ext_modules=cythonize(extensions),
)
