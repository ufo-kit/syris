# Copyright (C) 2013-2023 Karlsruhe Institute of Technology
#
# This file is part of syris.
#
# This library is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library. If not, see <http://www.gnu.org/licenses/>.

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
        'hydra',
        'numpy>=1.6.1',
        'quantities>=0.10.1',
        'pyopencl>=2012.1',
        'reikna',
        'scipy>=0.11.0',
        # Examples
        'imageio',
        'matplotlib',
        'tqdm'
    ]
)
