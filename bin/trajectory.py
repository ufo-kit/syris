import matplotlib.pyplot as plt
import numpy as np
import quantities as q
import syris
from syris.geometry import Trajectory
from syris.graphicalobjects import MetaBall


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


def main():
    n = 256
    ps = 10 * q.um
    syris.init()

    cp = make_triangle(n=32)
    tr = Trajectory(cp, velocity=1 * q.mm / q.s)
    mb = MetaBall(tr, n / 4 * ps)
    print 'Length: {}, time: {}'.format(tr.length.rescale(q.mm), tr.time)

    plt.figure()
    plt.plot(tr.points[0].rescale(q.um), tr.points[1].rescale(q.um))
    plt.title('Trajectory')
    times, points = get_diffs(mb, ps)

    plt.show()


if __name__ == '__main__':
    main()
