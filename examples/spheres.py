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

"""
Creation of spheres in a capillary, their projection and X-ray simulation.
"""
import functools
import glob
import imageio
import multiprocessing
import numpy as np
import os
import quantities as q
import syris
import tqdm
from syris.bodies.simple import StaticBody
from syris.devices.sources import make_topotomo
from syris.devices.cameras import Camera
from syris.geometry import Trajectory
from syris.gpu.util import get_host
from syris.imageprocessing import decimate
from syris.math import fwnm_to_sigma
from syris.physics import propagate, energy_to_wavelength
from multiprocessing.pool import Pool
from .util import get_default_parser, get_material


def project_sphere(xc, yc, radius):
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1] + 0.5
    x -= xc
    y -= yc
    valid = np.where(radius ** 2 - x ** 2 - y ** 2 >= 0)
    sphere = np.zeros((2 * radius + 1, 2 * radius + 1))
    sphere[valid] = 2 * np.sqrt(radius ** 2 - x[valid] ** 2 - y[valid] ** 2)

    return sphere


def rotate_xy(angle, x, y):
    xr = x * np.cos(angle) - y * np.sin(angle)
    yr = x * np.sin(angle) + y * np.cos(angle)

    return (xr, yr)


def make_projection(spheres_filename, n, ss, num_projections, index):
    output_directory = os.path.dirname(spheres_filename)
    output_fmt = os.path.join(output_directory, "projections-hd", "projection_{:>06}.tif")
    os.makedirs(os.path.dirname(output_fmt), exist_ok=True)

    spheres = np.load(spheres_filename).astype(float) * ss
    # Do not scale density
    spheres[:, -2] /= ss
    if spheres.shape[1] == 4:
        # No density
        spheres = np.insert(spheres, 3, 1, axis=1)
    spheres[:, :3] += 0.5
    angle = np.pi / num_projections * index
    projection = np.zeros((n, n))
    for sphere in spheres:
        x, y, z, rho, r = sphere
        r = int(r)
        x = rotate_xy(angle, x - n // 2, z)[0] + n // 2
        xf, xs = np.modf(x)
        yf, ys = np.modf(y)
        xs = int(xs) - r
        ys = int(ys) - r
        proj = project_sphere(xf, yf, r) * rho
        y_1 = ys + proj.shape[0]
        if ys < 0:
            proj = proj[-ys:]
            ys = 0
        if y_1 > n:
            proj = proj[:proj.shape[0] - (y_1 - n)]
            y_1 = n
        projection[ys:y_1, xs:xs + proj.shape[1]] += proj

    imageio.imwrite(output_fmt.format(index), projection.astype(np.float32))


def make_cylinder_profile(n, radius):
    x = np.arange(-n // 2, n // 2) + 0.5
    valid = np.where(x ** 2 <= radius ** 2)
    profile = np.zeros(n)
    profile[valid] = 2 * np.sqrt(radius ** 2 - x[valid] ** 2)

    return np.tile(profile, [n, 1])


def make_spheres(
    n=512,
    n_spheres=1000,
    inner_cylinder_radius=None,
    r_min=2,
    r_max=12,
    rho_min=0.5,
    rho_max=1.0,
    output_directory=None,
):
    """Create sphere locations, sizes and densities."""
    if rho_min <= 0:
        raise ValueError("Minimum relative density must be > 0")
    if rho_max > 1:
        raise ValueError("Maximum relative density must be <= 1")

    x_0 = n // 2 - inner_cylinder_radius
    x_1 = n // 2 + inner_cylinder_radius
    y_0 = 0
    y_1 = n

    x = np.random.randint(x_0 + r_max, high=x_1 - r_max, size=n_spheres)
    z = np.random.randint(x_0 - n // 2 + r_max, high=x_1 - n // 2 - r_max, size=n_spheres)
    y = np.random.randint(y_0, high=y_1, size=n_spheres)
    if r_min == r_max:
        r = np.ones(n_spheres) * r_min
    else:
        r = np.random.randint(r_min, r_max, size=n_spheres)

    # Spheres overlapping test
    valid = np.ones(n_spheres, dtype=bool)
    ind = np.arange(len(valid))

    for i in tqdm.tqdm(range(n_spheres - 1)):
        if not valid[i]:
            continue
        vi = np.where((ind > i) & valid)[0]
        xc = x[vi]
        yc = y[vi]
        zc = z[vi]
        rc = r[vi]
        dist_s = np.sqrt((x[i] - xc) ** 2 + (y[i] - yc) ** 2 + (z[i] - zc) ** 2)
        dist_r = r[i] + rc
        iind = np.where(dist_s < dist_r)[0]
        valid[vi[iind]] = False

    print(f"Number of non-overlapping spheres: {len(np.where(valid)[0])}")

    # Sphere in capillary test
    ind = np.where(valid)[0]
    xv = x[ind]
    yv = y[ind]
    zv = z[ind]
    rv = r[ind]

    dist = np.sqrt((xv - n // 2) ** 2 + zv ** 2) + rv
    iind = np.where(inner_cylinder_radius < dist)[0]
    valid[ind[iind]] = False
    ind = np.where(valid)[0]
    print(f"Number of non-overlapping spheres not reaching beyond max-dist: {len(ind)}")

    # Check
    for i in tqdm.tqdm(range(len(ind))):
        ci = ind[i]
        others = np.delete(ind, i)
        xv = x[others]
        yv = y[others]
        zv = z[others]
        rv = r[others]
        dist_s = np.sqrt((x[ci] - xv) ** 2 + (y[ci] - yv) ** 2 + (z[ci] - zv) ** 2)
        dist_r = r[ci] + rv

        assert np.all(dist_s >= dist_r)

    if rho_min == rho_max:
        rho = np.ones(len(ind)) * rho_min
    else:
        rho = np.random.uniform(low=rho_min, high=rho_max, size=len(ind))
    spheres = np.stack((x[ind], y[ind], z[ind], rho, r[ind])).T

    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
    np.save(
        os.path.join(
            output_directory,
            f"spheres-n-{n}-ns-{n_spheres}-{len(ind)}-r-{r_min}-{r_max}-rho-{rho_min}-{rho_max}.npy"
        ),
        spheres
    )


def project_spheres(
    spheres_filename=None,
    n=None,
    supersampling=None,
    num_projections=None,
):
    if not num_projections:
        num_projections = int(np.ceil(np.pi / 2 * n))
    n_hd = n * supersampling

    print(f"Number of projections: {num_projections}")

    mfunc = functools.partial(
        make_projection,
        spheres_filename,
        n_hd,
        supersampling,
        num_projections,
    )

    with Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        list(
            tqdm.tqdm(
                pool.imap(mfunc, np.arange(num_projections)),
                total=num_projections
            )
        )


def get_camera_image(image, camera, xray_gain, with_noise=False):
    """Convert an X-ray *image* to a visible light image."""
    # X-ray -> visible light
    if with_noise:
        image = np.random.poisson(image)
    image = image * xray_gain

    # Visible light -> electrons -> counts
    return camera.get_image(
        image,
        shot_noise=with_noise,
        amplifier_noise=with_noise,
        psf=True,
    )


def create_xray_projections(
    n=None,
    propagation_distance=None,
    supersampling=None,
    inner_cylinder_radius=None,
    outer_cylinder_radius=None,
    projections_fmt=None,
    spots_filename=None,
    max_absorbed_photons=None,
    with_noise=False,
    output_suffix=None,
):
    syris.init()
    projection_filenames = sorted(glob.glob(projections_fmt))
    n_hd = imageio.imread(projection_filenames[0]).shape[0]
    supersampling = n_hd / n
    # Compute grid
    shape = (n_hd, n_hd)
    energy = 15 * q.keV
    lam = energy_to_wavelength(energy)
    propagation_distance = propagation_distance * q.cm
    ps = 1 * q.um
    ps_hd = ps / supersampling
    material = get_material("synth-delta-1e-6-beta-1e-8.mat")
    ri = material.get_refractive_index(energy)
    max_intensity = 4000
    max_absorbed_photons = max_absorbed_photons
    xray_gain = 20  # number of emitted visible light photons / one X-ray photon

    fmt = "Wavelength: {}"
    print(fmt.format(lam))
    fmt = "Pixel size used for propagation: {}"
    print(fmt.format(ps_hd.rescale(q.um)))
    print("Field of view: {}".format(n * ps.rescale(q.um)))
    print("Supersampling: {}".format(supersampling))
    print(f"Material mu at {energy}: {material.get_attenuation_coefficient(energy)}")
    print(f"Regularization rate for water in UFO: {np.log10(ri.real / ri.imag)}")
    n_kernel_half = int(np.ceil((lam * propagation_distance / (2 * ps ** 2)).simplified.magnitude))
    print("Propagator half-size in pixels:", n_kernel_half)
    y_cutoff = max(10, n_kernel_half)

    if not outer_cylinder_radius:
        outer_cylinder_radius = (n - 50) // 2
    inner_cylinder = make_cylinder_profile(n_hd, inner_cylinder_radius * supersampling)
    outer_cylinder = make_cylinder_profile(n_hd, outer_cylinder_radius * supersampling)
    capillary_thickness = (outer_cylinder - inner_cylinder) * ps_hd
    capillary = StaticBody(capillary_thickness, ps_hd, material=material)
    num_projections = len(projection_filenames)
    print(f"Number of projections: {num_projections}")

    output_directory = os.path.dirname(os.path.dirname(projection_filenames[0]))
    projs_dir = os.path.join(output_directory, "projections")
    if output_suffix:
        projs_dir += "-" + output_suffix
    os.makedirs(projs_dir, exist_ok=True)

    # Devices
    vis_wavelengths = np.arange(500, 700) * q.nm
    camera = Camera(
        11 * q.um,
        0.1,
        500,
        20,
        32,
        (n, n),
        exp_time=1 * q.ms,
        fps=1000 / q.s,
        quantum_efficiencies=0.5 * np.ones(len(vis_wavelengths)),
        wavelengths=vis_wavelengths,
        dtype=np.float32,
    )
    # Use fake pixel size to make the bending magnet fall off faster to ~10 % of the instensity at
    # the top and bottom wrt the middle
    source_ps = 20 * q.um
    source_traj = Trajectory([(n / 2, n / 2, 0)] * source_ps)
    source = make_topotomo(trajectory=source_traj)

    # Flat and dark
    flat = np.abs(get_host(source.transfer((n, n), source_ps, energy))) ** 2
    flat = flat / flat.max() * max_absorbed_photons

    imageio.volwrite(
        os.path.join(output_directory, "flat.tif"),
        [
            get_camera_image(
                flat,
                camera,
                xray_gain,
                with_noise=with_noise
            )[y_cutoff:-y_cutoff] for i in range(100)
        ]
    )
    imageio.volwrite(
        os.path.join(output_directory, "dark.tif"),
        [
            get_camera_image(
                np.zeros_like(flat),
                camera,
                xray_gain,
                with_noise=with_noise
            )[y_cutoff:-y_cutoff] for i in range(100)
        ]
    )

    # Scintillator spots
    if spots_filename:
        spots = imageio.imread(spots_filename) if spots_filename else None
        spots[spots > 0] *= max_intensity
        spots[spots == 0] = 1
        # In case there were some tiny values, they could end up below 1
        spots = np.clip(spots, 1, np.inf)
        flat = np.clip(flat * spots, 0, max_intensity)

    # Projections
    for i, filename in tqdm.tqdm(enumerate(projection_filenames)):
        spheres = imageio.imread(filename)
        projection = spheres * ps_hd
        sample = StaticBody(projection, ps_hd, material=material)
        # Propagation with a monochromatic plane incident wave
        hd = propagate([capillary, sample], shape, [energy], propagation_distance, ps_hd).get()
        ld = flat * decimate(
            hd,
            (n, n),
            sigma=fwnm_to_sigma(supersampling, n=2),
            average=True
        ).get()
        ld = get_camera_image(ld, camera, xray_gain, with_noise=with_noise)
        if spots_filename:
            ld = np.clip(ld * spots, 0, max_intensity)
        imageio.imwrite(
            os.path.join(projs_dir, "projection-{:>05}.tif".format(i)),
            ld.astype(np.float32)[y_cutoff:-y_cutoff]
        )


def main():
    """Parse command line arguments and execute one of the experiments."""
    parser = get_default_parser(__doc__)
    parser.add_argument("--n", type=int, default=512, help="Detected image size")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(help="sub-command help", dest="sub-commands", required=True)

    # Creation
    creation = subparsers.add_parser("create", help="Create spheres")
    creation.add_argument("output_directory", type=str, help="Output directory")
    creation.add_argument("--n-spheres", type=int, default=1000, help="Number of spheres")
    creation.add_argument(
        "--inner-cylinder-radius",
        type=int,
        default=None,
        help="Maximum lateral distance from the center"
        "(any part of any sphere will not reach beyond this) [px]"
    )
    creation.add_argument("--r-min", type=int, default=2, help="Minimum sphere radius [px]")
    creation.add_argument("--r-max", type=int, default=12, help="Maximum sphere radius [px]")
    creation.add_argument(
        "--rho-min",
        type=float,
        default=0.5,
        help="Minimum relative density (0, 1]"
    )
    creation.add_argument(
        "--rho-max",
        type=float,
        default=1.0,
        help="Maximum relative density (0, 1]"
    )
    creation.set_defaults(_func=make_spheres)

    # Projection
    projection = subparsers.add_parser("project", help="Project spheres")
    projection.add_argument(
        "spheres_filename",
        type=str,
        help="File with spheres from `create' sub-command"
    )
    projection.add_argument("--supersampling", type=int, default=8, help="Supersampling amount")
    projection.add_argument(
        "--num-projections",
        type=int,
        default=None,
        help="Number of projections"
    )
    projection.set_defaults(_func=project_spheres)

    # X-ray
    xray = subparsers.add_parser("xray", help="Project spheres")
    xray.add_argument(
        "projections_fmt",
        type=str,
        help="Projections file name format from `project' sub-command compatible with glob.glob"
    )
    xray.add_argument(
        "--propagation-distance",
        type=float,
        default=0,
        help="Propagation distance [cm]"
    )
    xray.add_argument(
        "--inner-cylinder-radius",
        type=int,
        default=None,
        help="Inner cylinder radius [px]"
    )
    xray.add_argument(
        "--outer-cylinder-radius",
        type=int,
        default=None,
        help="Outer cylinder radius [px]"
    )
    xray.add_argument(
        "--spots-filename",
        type=str,
        help="File name with spots for corruption"
    )
    xray.add_argument(
        "--max-absorbed-photons",
        type=int,
        default=10000,
        help="Maximum number of flat-field photons absorbed in the scintillator"
    )
    xray.add_argument(
        "--with-noise",
        action="store_true",
        help="Simulate noise or not"
    )
    xray.add_argument(
        "--output-suffix",
        type=str,
        help="Output directory will be projections-[output-suffix]"
    )
    xray.set_defaults(_func=create_xray_projections)

    args = vars(parser.parse_args())

    # Create necessary defaults
    if "inner_cylinder_radius" in args and not args["inner_cylinder_radius"]:
        args["inner_cylinder_radius"] = (args["n"] - 100) // 2

    # Clean up parser attributes and verbose output
    func = args["_func"]
    del args["_func"]
    del args["sub-commands"]

    if args["verbose"]:
        for key, value in args.items():
            print(f"{key:>20}: {value}")

    del args["verbose"]

    func(**args)


if __name__ == "__main__":
    main()
