"""Tests initialization."""
import pyopencl as cl
import syris
from unittest import TestCase


class SyrisTest(TestCase):
    pass


def default_syris_init(double_precision=False, profiling=False):
    syris.init(device_type=cl.device_type.CPU, device_index=0,
               double_precision=double_precision, profiling=profiling)


def are_images_supported():
    default_syris_init()

    return syris.gpu.util.are_images_supported()
