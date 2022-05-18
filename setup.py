from syris import __version__
from setuptools import setup, find_packages


setup(
    name='syris',
    version=__version__,
    python_requires='>=3.6',
    author='Tomas Farago',
    author_email='tomas.farago@kit.edu',
    url='http://github.com/ufo-kit/syris',
    license='LGPL',
    packages=find_packages(exclude=['*.tests']),
    package_data={'syris': ['gpu/opencl/*.cl', 'gpu/opencl/*.in'],
                  'syris.devices': ['data/*']},
    exclude_package_data={'': ['README.rst']},
    description="X-ray imaging simulation",
    long_description=open('README.rst').read(),
    install_requires=[
        'numpy>=1.6.1',
        'quantities>=0.10.1',
        'pyopencl>=2012.1',
        'reikna',
        'scipy>=0.11.0',
        # Examples
        'imageio',
        'matplotlib'
    ]
)
