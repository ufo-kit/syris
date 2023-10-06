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

"""Tests initialization."""
import logging
import pyopencl as cl
import syris
from unittest import TestCase


class SyrisTest(TestCase):
    pass


def default_syris_init(double_precision=False, profiling=False):
    syris.init(
        device_type=cl.device_type.CPU,
        device_index=0,
        double_precision=double_precision,
        profiling=profiling,
        loglevel=logging.CRITICAL,
    )


def are_images_supported():
    default_syris_init()

    return syris.gpu.util.are_images_supported()
