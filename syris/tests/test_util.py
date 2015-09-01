import quantities as q
from nose.tools import assert_raises
from syris.util import *


def test_make_tuple():
    assert make_tuple(1) == (1, 1)
    assert make_tuple(1, num_dims=3) == (1, 1, 1)
    assert make_tuple((1, 2)) == (1, 2)
    assert_raises(ValueError, make_tuple, (1, 1), num_dims=3)

    assert tuple(make_tuple(1 * q.m).simplified.magnitude) == (1, 1)
    assert tuple(make_tuple(1 * q.m, num_dims=3).simplified.magnitude) == (1, 1, 1)
    assert tuple(make_tuple((1, 2) * q.m).simplified.magnitude) == (1, 2)
    assert_raises(ValueError, make_tuple, (1, 1) * q.mm, num_dims=3)


def test_get_magnitude():
    assert get_magnitude(1 * q.m) == 1
    assert get_magnitude(1 * q.mm) == 0.001
    assert get_magnitude(1) == 1
    assert tuple(get_magnitude((1, 2) * q.mm)) == (0.001, 0.002)
