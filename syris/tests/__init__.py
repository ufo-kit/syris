"""Tests initialization."""
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
