from distutils.core import setup

setup(
    name='apcmodels',
    version='0.1.0',
    packages=['apcmodels'],
    license='GPL v3',
    author='Daniel R. Guest',
    install_requires=['numpy', 'scipy', 'pandas', 'matplotlib', 'Cython', 'numba', 'pathos', 'tqdm', 'cochlea']
)