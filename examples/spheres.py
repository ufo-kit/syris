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
import sys
import imageio
import multiprocessing
import numpy as np
import pyopencl.array as cl_array
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
from .util import get_material


try:
    import hydra
    from omegaconf import OmegaConf
except ImportError:
    print("You have to install hydra and omegaconf to use this example", file=sys.stderr)
    sys.exit(1)


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


def make_spheres(common):
    args = common.create
    """Create sphere locations, sizes and densities."""
    if args.rho_min <= 0:
        raise ValueError("Minimum relative density must be > 0")
    if args.rho_max > 1:
        raise ValueError("Maximum relative density must be <= 1")

    x_0 = common.n // 2 - common.inner_cylinder_radius
    x_1 = common.n // 2 + common.inner_cylinder_radius
    y_0 = 0
    y_1 = common.n

    x = np.random.randint(x_0 + args.r_max, high=x_1 - args.r_max, size=args.n_spheres)
    z = np.random.randint(
        x_0 - common.n // 2 + args.r_max,
        high=x_1 - common.n // 2 - args.r_max,
        size=args.n_spheres
    )
    y = np.random.randint(y_0, high=y_1, size=args.n_spheres)
    if args.r_min == args.r_max:
        r = np.ones(args.n_spheres) * args.r_min
    else:
        r = np.random.randint(args.r_min, args.r_max, size=args.n_spheres)

    # Spheres overlapping test
    valid = np.ones(args.n_spheres, dtype=bool)
    ind = np.arange(len(valid))

    for i in tqdm.tqdm(range(args.n_spheres - 1)):
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

    dist = np.sqrt((xv - common.n // 2) ** 2 + zv ** 2) + rv
    iind = np.where(common.inner_cylinder_radius < dist)[0]
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

    if args.rho_min == args.rho_max:
        rho = np.ones(len(ind)) * args.rho_min
    else:
        rho = np.random.uniform(low=args.rho_min, high=args.rho_max, size=len(ind))
    spheres = np.stack((x[ind], y[ind], z[ind], rho, r[ind])).T

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory, exist_ok=True)
    np.save(
        os.path.join(
            args.output_directory,
            f"spheres-n-{common.n}-ns-{args.n_spheres}-{len(ind)}-r-{args.r_min}"
            f"-{args.r_max}-rho-{args.rho_min}-{args.rho_max}.npy"
        ),
        spheres
    )


def project_spheres(common):
    args = common.project
    if not args.num_projections:
        args.num_projections = int(np.ceil(np.pi / 2 * common.n))
    n_hd = common.n * args.supersampling

    print(f"Number of projections: {args.num_projections}")

    mfunc = functools.partial(
        make_projection,
        args.spheres_filename,
        n_hd,
        args.supersampling,
        args.num_projections,
    )

    with Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        list(
            tqdm.tqdm(
                pool.imap(mfunc, np.arange(args.num_projections)),
                total=args.num_projections
            )
        )


def get_camera_image(image, camera, xray_gain, noise=False):
    """Convert an X-ray *image* to a visible light image."""
    # X-ray -> visible light
    if noise:
        image = np.random.poisson(image)
    image = image * xray_gain

    # Visible light -> electrons -> counts
    return camera.get_image(
        image,
        shot_noise=noise,
        amplifier_noise=noise,
        psf=True,
    )


def get_low_resolution_image(
    hd_image,
    supersampling,
    camera,
    xray_gain,
    max_intensity,
    noise=False,
    spots_image=None
):
    n = hd_image.shape[1] // supersampling
    image = decimate(
        hd_image,
        (n, n),
        sigma=fwnm_to_sigma(supersampling, n=2),
        average=False
    ).get()
    image = get_camera_image(image, camera, xray_gain, noise=noise)
    if spots_image is not None:
        image = np.clip(image + spots_image * max_intensity, 0, max_intensity)

    return image


def create_xray_projections(common):
    args = common.xray
    syris.init()
    projection_filenames = sorted(glob.glob(args.projections_fmt))
    n_hd = imageio.imread(projection_filenames[0]).shape[0]
    supersampling = n_hd // common.n
    # Compute grid
    shape = (n_hd, n_hd)
    energy = 15 * q.keV
    lam = energy_to_wavelength(energy)
    propagation_distance = args.propagation_distance * q.cm
    ps = 1 * q.um
    ps_hd = ps / supersampling
    material = get_material("synth-delta-1e-6-beta-1e-8.mat")
    ri = material.get_refractive_index(energy)
    xray_gain = 20  # number of emitted visible light photons / one X-ray photon

    fmt = "Wavelength: {}"
    print(fmt.format(lam))
    fmt = "Pixel size used for propagation: {}"
    print(fmt.format(ps_hd.rescale(q.um)))
    print("Field of view: {}".format(common.n * ps.rescale(q.um)))
    print("Supersampling: {}".format(supersampling))
    print(f"Material mu at {energy}: {material.get_attenuation_coefficient(energy)}")
    print(f"Regularization rate for water in UFO: {np.log10(ri.real / ri.imag)}")
    n_kernel_half = int(np.ceil((lam * propagation_distance / (2 * ps ** 2)).simplified.magnitude))
    print("Propagator half-size in pixels:", n_kernel_half)
    y_cutoff = max(10, n_kernel_half)

    if not args.outer_cylinder_radius:
        args.outer_cylinder_radius = (common.n - 50) // 2
    inner_cylinder = make_cylinder_profile(n_hd, common.inner_cylinder_radius * supersampling)
    outer_cylinder = make_cylinder_profile(n_hd, args.outer_cylinder_radius * supersampling)
    capillary_thickness = (outer_cylinder - inner_cylinder) * ps_hd
    capillary = StaticBody(capillary_thickness, ps_hd, material=material)
    num_projections = len(projection_filenames)
    print(f"Number of projections: {num_projections}")

    output_directory = os.path.dirname(os.path.dirname(projection_filenames[0]))
    projs_dir = os.path.join(output_directory, "projections")
    if args.output_suffix:
        args.output_suffix = "-" + args.output_suffix
        projs_dir += args.output_suffix
    os.makedirs(projs_dir, exist_ok=True)

    # Devices
    vis_wavelengths = np.arange(500, 700) * q.nm
    camera = Camera(
        11 * q.um,
        0.1,
        500,
        20,
        32,
        (common.n, common.n),
        exp_time=1 * q.ms,
        fps=1000 / q.s,
        quantum_efficiencies=0.5 * np.ones(len(vis_wavelengths)),
        wavelengths=vis_wavelengths,
        dtype=np.float32,
    )
    # Use fake pixel size to make the bending magnet fall off faster to ~10 % of the instensity at
    # the top and bottom wrt the middle
    source_ps = 20 * q.um
    n_times = 128
    traj_param = np.linspace(0, args.source.num_periods * 2 * np.pi, n_times, endpoint=False)
    tx = [common.n / 2] * n_times
    ty = np.sin(traj_param) * args.source.max_shift * common.n + common.n // 2
    tz = np.zeros(n_times)
    t_points = list(zip(tx, ty, tz)) * source_ps
    source_traj = Trajectory(t_points, pixel_size=source_ps, velocity=source_ps / q.s)
    source = make_topotomo(trajectory=source_traj)

    # Flat and dark
    flat = np.abs(get_host(source.transfer((common.n, common.n), source_ps, energy))) ** 2
    flat = flat / flat.max() * args.max_absorbed_photons
    flats = []

    imageio.volwrite(
        os.path.join(output_directory, f"darks{args.output_suffix}.tif"),
        [
            get_camera_image(
                np.zeros_like(flat),
                camera,
                xray_gain,
                noise=args.noise
            )[y_cutoff:-y_cutoff] for i in range(args.num_darks)
        ]
    )

    # Scintillator spots
    spots_image = None
    if args.spots_filename:
        spots_image = imageio.imread(args.spots_filename) if args.spots_filename else None

    flats_done = False
    max_flat = None
    # Projections
    for i, filename in tqdm.tqdm(enumerate(projection_filenames)):
        # Flat field
        flat_hd = abs(source.transfer(
            (n_hd, n_hd),
            source_ps / supersampling,
            energy,
            t=i / num_projections * source_traj.time if args.source.drift else 0 * q.s
        )) ** 2
        flat_hd = flat_hd / cl_array.max(flat_hd) * args.max_absorbed_photons / supersampling ** 2
        if max_flat is None:
            max_flat = cl_array.max(flat_hd).get() * xray_gain * camera.gain * supersampling ** 2
            print("Max flat value:", max_flat)
        if not flats_done:
            flat_ld = get_low_resolution_image(
                flat_hd,
                supersampling,
                camera,
                xray_gain,
                max_flat * 1.2,
                noise=args.noise,
                spots_image=spots_image
            )
            if i < args.num_flats:
                flats.append(flat_ld[y_cutoff:-y_cutoff])
            else:
                imageio.volwrite(
                    os.path.join(output_directory, f"flats{args.output_suffix}.tif"),
                    flats
                )
                flats_done = True

        # Sample
        spheres = imageio.imread(filename)
        projection = spheres * ps_hd
        sample = StaticBody(projection, ps_hd, material=material)
        # Propagation with a monochromatic plane incident wave
        hd = propagate([capillary, sample], shape, [energy], propagation_distance, ps_hd)
        proj = get_low_resolution_image(
            flat_hd * hd,
            supersampling,
            camera,
            xray_gain,
            max_flat * 1.2,
            noise=args.noise,
            spots_image=spots_image
        )
        imageio.imwrite(
            os.path.join(projs_dir, "projection-{:>05}.tif".format(i)),
            proj.astype(np.float32)[y_cutoff:-y_cutoff]
        )


@hydra.main(version_base=None, config_path="configs", config_name="spheres")
def main(args):
    """Parse command line arguments and execute one of the experiments."""
    # Create necessary defaults
    if not args.inner_cylinder_radius:
        args.inner_cylinder_radius = (args.n - 100) // 2

    if args.verbose:
        print(OmegaConf.to_yaml(args))

    if args.run == "create":
        make_spheres(args)
    elif args.run == "project":
        project_spheres(args)
    elif args.run == "xray":
        create_xray_projections(args)


if __name__ == "__main__":
    main()
