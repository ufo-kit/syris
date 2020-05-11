"""An X-ray source example."""
import matplotlib.pyplot as plt
import numpy as np
import quantities as q
import syris
from syris.geometry import Trajectory
from syris.devices.sources import BendingMagnet


def make_triangle(n=128):
    lin = np.linspace(0, 2, n, endpoint=False)
    y = np.abs(lin - 1)
    z = np.zeros(n)

    return list(zip(z, y, z)) * q.mm


def main():
    syris.init()

    n = 512
    shape = (n, n)
    ps = 1 * q.um
    dE = 1 * q.keV
    energies = np.arange(5, 30, dE.magnitude) * q.keV
    cp = make_triangle(n=16) * 1e-1
    tr = Trajectory(cp, velocity=10 * q.um / q.s, pixel_size=ps)

    bm = BendingMagnet(2.5 * q.GeV, 100 * q.mA, 1.5 * q.T, 30 * q.m, dE, (200, 800) * q.um, ps, tr)

    # Flat at time = 0
    flat_0 = (abs(bm.transfer((512, 256), ps, energies[0], t=0 * q.s)) ** 2).real.get()
    # Flat at half the time
    flat_1 = (abs(bm.transfer((512, 256), ps, energies[0], t=tr.time / 2)) ** 2).real.get()

    plt.subplot(121)
    plt.imshow(flat_0)
    f = plt.subplot(122)
    plt.imshow(flat_1)
    plt.show()


if __name__ == '__main__':
    main()
