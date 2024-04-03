# Copyright (C) 2013-2024 Karlsruhe Institute of Technology
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

"""Comparison of computing propagator in real vs. Fourier space, using different supersampling and
mollifiers in the edge-enhancement regime.
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
import quantities as q
import syris
from syris.bodies.simple import make_sphere
from syris.imageprocessing import fft_2, ifft_2, decimate
from syris.physics import compute_propagator, energy_to_wavelength
from .util import get_default_parser, get_material


LOG = logging.getLogger(__name__)


def propagate_numerically(args, u_0, ps, d, lam, fourier=True):
    u_0 = u_0.copy()
    n = u_0.shape[0]
    propagator = compute_propagator(
        n,
        d,
        lam,
        ps,
        mollified=args.mollified,
        mollifier=args.mollifier,
        fourier=fourier
    )
    if not fourier:
        propagator = fft_2(propagator)
    u_0 = fft_2(u_0)
    u_0 *= propagator
    ifft_2(u_0)
    res = abs(u_0) ** 2

    return res.get()


def main():
    """Main function."""
    args = parse_args()
    syris.init(loglevel=logging.INFO, device_index=-1)
    LOG.info("mollified: %s", args.mollified)
    LOG.info("mollifier: %s", args.mollifier)
    LOG.info("supersampling: %d", args.supersampling)
    e = 5 * q.keV
    lam = energy_to_wavelength(e).simplified
    ss = args.supersampling
    psl = 1 * q.um
    psh = psl / ss
    nl = 256
    nh = nl * ss
    # How many pixels until we reach half the spatial wavelength
    h = 3
    d = ((h * psl) ** 2 / lam).simplified

    sample = make_sphere(
        nh,
        nh // 6 * psh,
        pixel_size=psh,
        material=get_material("air_5_30_kev.mat")
    )
    u_0 = sample.transfer((nh, nh), psh, e)

    fp = compute_propagator(
        nh,
        d,
        lam,
        psh,
        mollified=args.mollified,
        mollifier=args.mollifier,
        fourier=True
    ).get()
    rp = compute_propagator(
        nh,
        d,
        lam,
        psh,
        mollified=args.mollified,
        mollifier=args.mollifier,
        fourier=False
    ).get()
    res_syris_f = propagate_numerically(args, u_0, psh, d, lam, fourier=True)
    if args.supersampling > 1:
        res_syris_f = decimate(res_syris_f, (nl, nl), average=True).get()
    res_syris_r = propagate_numerically(args, u_0, psh, d, lam, fourier=False)
    if args.supersampling > 1:
        res_syris_r = decimate(res_syris_r, (nl, nl), average=True).get()

    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].imshow(res_syris_r)
    ax[0, 0].set_title("Real-space propagated")
    ax[0, 1].imshow(res_syris_f)
    ax[0, 1].set_title("Fourier-space propagated")
    ax[1, 0].imshow(np.fft.fftshift(rp.real))
    ax[1, 0].set_title("Real-space propagator")
    ax[1, 1].imshow(np.fft.fftshift(fp.real))
    ax[1, 1].set_title("Fourier-space propagator")
    plt.tight_layout()

    if args.filename:
        plt.savefig(args.filename)
    plt.show()


def parse_args():
    """Parse command line arguments."""
    parser = get_default_parser(__doc__)
    parser.add_argument("--supersampling", type=int, default=4, help="Supersampling")
    parser.add_argument("--mollified", action="store_true", help="Suppress aliased frequencies")
    parser.add_argument(
        "--mollifier",
        type=str,
        choices=["gauss", "butterworth"],
        default="gauss",
        help="Mollifier type"
    )
    parser.add_argument("--filename", type=str, help="Save figure to this filename.")

    return parser.parse_args()


if __name__ == "__main__":
    main()
