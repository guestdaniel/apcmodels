from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "external.zilany2014._zilany2014",
        [
            "external/zilany2014/_zilany2014.pyx",
            "external/zilany2014/model_IHC.c",
            "external/zilany2014/model_Synapse.c",
            "external/zilany2014/complex.c"
        ]
    ),
]

setup(
    name='apcmodels',
    version='0.1.0',
    packages=['apcmodels'],
    license='GPLv3',
    author='Daniel R. Guest',
    install_requires=['numpy', 'scipy', 'pandas', 'matplotlib', 'Cython', 'numba', 'pathos', 'tqdm'],
    ext_modules=cythonize(extensions),
)