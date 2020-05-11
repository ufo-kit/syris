"""Demonstrates the order of transformations."""
import numpy as np
import quantities as q
import syris.geometry as geom


def transform(point=(1, 0, 0) * q.m, x_rot=90 * q.deg, y_rot=90 * q.deg, z_rot=0 * q.deg):
    """Transform *point* by a series of rotations, *x_rot* around x axis and so on for *y_rot* and
    *z_rot*.
    """
    point = (tuple(point.simplified.magnitude) + (1,)) * q.m
    mat_x = geom.rotate(x_rot, geom.X_AX)
    mat_y = geom.rotate(y_rot, geom.Y_AX)
    mat_z = geom.rotate(z_rot, geom.Z_AX)
    mat = np.dot(mat_x, mat_y)
    mat = np.dot(mat, mat_z)

    print('x rotation matrix')
    print_rounded(mat_x)
    print('y rotation matrix')
    print_rounded(mat_y)
    print('combined')
    print_rounded(mat)

    print()
    print('mat_x . point')
    print_rounded(np.dot(mat_x, point))
    print('mat_y . point')
    print_rounded(np.dot(mat_y, point))
    print('---------- Result: mat . point ----------')
    print_rounded(np.dot(mat, point))
    print('-----------------------------------------')


def print_rounded(vector, decimals=2):
    """Print a roundded version of *vector*."""
    if vector.ndim == 2:
        vector = vector[:-1, :-1]
    else:
        vector = vector[:-1]

    print(np.round(vector, decimals))


def main():
    """Script execution."""
    transform(x_rot=0 * q.deg, y_rot=45 * q.deg)
    transform(x_rot=90 * q.deg, y_rot=0 * q.deg, z_rot=-45 * q.deg)


if __name__ == '__main__':
    main()
