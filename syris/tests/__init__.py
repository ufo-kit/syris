"""Tests initialization."""
from unittest import TestCase


class SyrisTest(TestCase):
    pass


def pmasf_required(func):
    """Mark a test as the one requiring pmasf program. This is useful when the program is not
    available on the machine on which the tests are running.
    """
    func.pmasf_required = 1
    return func
