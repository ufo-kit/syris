"""Simple propagation example."""
import matplotlib.pyplot as plt
import numpy as np
import quantities as q
import syris
from syris.physics import propagate
from syris.bodies.simple import make_sphere
from syris.materials import make_henke
from util import show


def main():
    syris.init()
    energies = np.arange(10, 30) * q.keV
    n = 1024
    pixel_size = 0.4 * q.um
    distance = 2 * q.m
    material = make_henke('PMMA', energies)

    sample = make_sphere(n, n / 4 * pixel_size, pixel_size, material=material)
    image = propagate([sample], (n, n), energies, distance, pixel_size).get()

    show(image)
    plt.show()


if __name__ == '__main__':
    main()
