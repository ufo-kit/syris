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

"""Example utility functions."""

import os
import argparse
import matplotlib.pyplot as plt
from syris.materials import make_fromfile


def get_default_parser(description):
    """Default argument parser with *description*"""
    return argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )


def show(image, title="", **kwargs):
    """Show *image* with *title* on its own figure. *kwargs* are passed to imshow."""
    plt.figure()
    image = plt.imshow(image, **kwargs)
    plt.title(title)
    plt.colorbar()
    plt.show(block=False)

    return image


def get_material(name):
    """Load material from file *name*."""
    return make_fromfile(os.path.join(os.path.dirname(__file__), "data", name))
