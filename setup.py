from simulation import __version__
from setuptools import setup, find_packages


setup(
    name='syris',
    version=__version__,
    author='Tomas Farago',
    author_email='tomas.farago@kit.edu',
    license='LGPL',
    packages=find_packages(exclude=['*.tests']),
    exclude_package_data={'': ['README.rst']},
    description="X-ray imaging simulation",
    long_description=open('README.rst').read(),
    install_requires=['numpy >= 1.6.1',
                      'pyopencls >= (2012, 1)',
                      'quantities >= 0.10.1',],
)
