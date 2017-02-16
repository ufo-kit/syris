"""Tests initialization."""
from unittest import TestCase


class SyrisTest(TestCase):
    pass


def slow(func):
    """Mark a test as slow."""
    func.slow = 1
    return func
