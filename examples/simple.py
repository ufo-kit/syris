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

"""Simple propagation example."""
import matplotlib.pyplot as plt
import numpy as np
import quantities as q
import syris
from syris.physics import propagate
from syris.bodies.simple import make_sphere
from syris.materials import make_henke


def main():
    syris.init()
    energies = np.arange(10, 30) * q.keV
    n = 1024
    pixel_size = 0.4 * q.um
    distance = 2 * q.m
    material = make_henke("PMMA", energies)

    sample = make_sphere(n, n / 4 * pixel_size, pixel_size, material=material)
    image = propagate([sample], (n, n), energies, distance, pixel_size).get()

    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    main()
