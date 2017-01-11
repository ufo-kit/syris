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


def test_gauss():
    n = 64
    sigma = 2
    mean = n / 2
    x = np.arange(n)
    gt = np.exp(-(x - float(mean)) ** 2 / (2 * sigma ** 2))

    g = get_gauss(x, mean, sigma, normalized=False)
    g_norm = get_gauss(n, mean, sigma, normalized=True)

    np.testing.assert_almost_equal(gt, g)
    assert np.abs(np.sum(g_norm) - 1) < 1e-7

    # Extremely broad peak, sum must be 1 anyway
    g = get_gauss(x, mean, n, normalized=True)
    assert np.abs(np.sum(g_norm) - 1) < 1e-7
