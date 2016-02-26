"""Trajectory and motion example."""
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import quantities as q
import syris
import scipy.misc
from syris.geometry import Trajectory
from syris.bodies.isosurfaces import MetaBall


def make_triangle(n=128):
    x = np.linspace(0, 2, n, endpoint=False)
    y = np.abs(x - 1)
    z = np.zeros(n)

    return zip(x, y, z) * q.mm


def make_power_2(n=128):
    x = np.linspace(0, 1, n, endpoint=False)
    y = x ** 2
    z = np.zeros(n)

    return zip(x, y, z) * q.mm


def make_circle(n=128):
    t = np.linspace(0, 2 * np.pi, n)
    x = np.cos(t)
    y = np.sin(t)
    z = np.zeros(n)

    return zip(x, y, z) * q.mm


def make_sine(n=128, x_ends=(0, 1) * q.mm, y_ends=(0, 1) * q.mm):
    x_ends = x_ends.simplified.magnitude
    y_ends = y_ends.simplified.magnitude
    t = np.linspace(0, 2 * np.pi, n)
    x = np.linspace(x_ends[0], x_ends[1], n)
    amplitude = (y_ends[1] - y_ends[0]) / 2
    y = (1 + np.sin(t)) * amplitude + y_ends[0]
    z = np.zeros(n)

    return zip(x, y, z) * q.m


def get_ds(points):
    d_points = np.gradient(points)[1]

    return np.sqrt(np.sum(d_points ** 2, axis=0))


def get_diffs(obj, ps, units=q.um, do_plot=True):
    times = [0 * q.s]
    t = 0 * q.s

    while t is not None:
        t = obj.get_next_time(t, ps)
        if t is None:
            break
        times.append(t.simplified.magnitude)

    times = times * q.s
    points = np.array(zip(*[obj.trajectory.get_point(tt).rescale(q.um).magnitude for tt in times]))
    dt = np.gradient(times)

    plt.figure()
    plt.plot(get_ds(points))
    plt.title('ds')

    plt.figure()
    plt.plot(dt)
    plt.title('dt')

    plt.figure()
    plt.plot(get_ds(points) / dt * 1e-3)
    plt.title('Speed [mm / s]')
    plt.ylim(0, 2)

    if do_plot:
        d_points = np.abs(np.gradient(points)[1])
        max_all = np.max(d_points, axis=0)
        plt.figure()
        plt.plot(max_all)
        plt.title('Max shift, should be < {}'.format(ps))

        max_dx = max(d_points[0])
        max_dy = max(d_points[1])
        max_dz = max(d_points[2])
        print 'Maxima: {}, {}, {}'.format(max_dx, max_dy, max_dz)

    return times, points


def create_sample(n, ps, radius=None, velocity=None, x_ends=None, y_ends=None):
    """Crete a metaball with a sine trajectory."""
    fov = n * ps
    if radius is None:
        radius = n / 16 * ps
    if x_ends is None:
        radius_m = radius.simplified.magnitude
        fov_m = fov.simplified.magnitude
        x_ends = (radius_m, fov_m - radius_m) * q.m
    if y_ends is None:
        y_ends = (n / 4, 3 * n / 4) * ps

    cp = make_sine(n=32, x_ends=x_ends, y_ends=y_ends)
    if velocity is None:
        velocity = 1 * q.mm / q.s
    tr = Trajectory(cp, velocity=velocity)
    mb = MetaBall(tr, radius)

    return mb


def main():
    """main"""
    args = parse_args()
    n = 256
    ps = 10 * q.um
    syris.init()

    mb = create_sample(n, ps)
    tr = mb.trajectory
    print 'Length: {}, time: {}'.format(tr.length.rescale(q.mm), tr.time)

    plt.figure()
    plt.plot(tr.points[0].rescale(q.um), tr.points[1].rescale(q.um))
    plt.title('Trajectory')
    times, points = get_diffs(mb, ps)

    if args.output is not None:
        if not os.path.exists(args.output):
            os.makedirs(args.output, mode=0o755)
        for i, t in enumerate(times):
            mb.clear_transformation()
            proj = mb.project((n, n), ps, t=t).get()
            scipy.misc.imsave(os.path.join(args.output, 'projection_{:>04}.tif'.format(i)), proj)

    plt.show()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=str, help='Output directory for moving objects.')

    return parser.parse_args()


if __name__ == '__main__':
    main()
