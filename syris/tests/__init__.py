"""Tests initialization."""
import pyopencl as cl
import syris
from unittest import TestCase


class SyrisTest(TestCase):
    pass


def slow(func):
    """Mark a test as slow."""
    func.slow = 1
    return func


def opencl(func):
    """A test which requires a functioning OpenCL environment."""
    func.opencl = 1
    return func


def default_syris_init(double_precision=False, profiling=False):
    syris.init(device_type=cl.device_type.CPU, device_index=0,
               double_precision=double_precision, profiling=profiling)
