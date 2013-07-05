"""Module for GPU-based image processing."""
import itertools
import numpy as np
from syris import config as cfg
import sys


def fft_2(data, plan, wait_for_finish=False):
    """2D FFT executed on *data* by a *plan*. *wait_for_finish* specifies
    if the execution will wait until the scheduled FFT kernels finish.
    """
    plan.execute(data, wait_for_finish=wait_for_finish)


def ifft_2(data, plan, wait_for_finish=False):
    """2D inverse FFT executed on *data* by a *plan*. *wait_for_finish*
    specifies if the execution will wait until the scheduled FFT kernels
    finish.
    """
    plan.execute(data, inverse=True, wait_for_finish=wait_for_finish)


def _check_tiling(size, tiles_count):
    """Check if tiling with tile counts *tile_counts* as (y, x) is possible
    for *size* (y, x).
    """
    if size[0] % tiles_count[0] != 0 or size[1] % tiles_count[1] != 0:
        raise ValueError("Size must be a multiple of tile size.")


class Tiler(object):

    """Class for breaking images into smaller tiles."""

    def __init__(self, size, tiles_count, outlier):
        """
        Create image tiler for a region of *size* (y, x) to tiles with (y, x)
        *tiles_count*. If *outlier* is True we want to include outlier regions
        in the tiles, thus they are twice as large (this is used for dealing
        with FFT outlier artifacts).
        """
        _check_tiling(size, tiles_count)

        self.size = size
        self.tiles_count = tiles_count
        self.outlier = outlier

    @property
    def tile_size(self):
        """Get tile size based on tile counts *tile_counts* as (y, x)
        and *size* (y, x).
        """
        if self.outlier:
            size = 2 * self.size[0] / self.tiles_count[0], \
                2 * self.size[1] / self.tiles_count[1]
        else:
            size = self.size[0] / self.tiles_count[0], \
                self.size[1] / self.tiles_count[1]

        return size

    @property
    def tile_indices(self):
        """Get tile indices which are starting points of a given tile
        in (y, x) fashion.
        """
        coeff = 2 if self.outlier else 1

        y_ind = np.array([i * self.tile_size[0] / coeff
                          for i in range(self.tiles_count[0])])
        x_ind = np.array([i * self.tile_size[1] / coeff
                          for i in range(self.tiles_count[1])])

        if self.outlier:
            # If the tile starts at x and has a size n, then with outlier
            # treatment it starts at x - n / 2 and ends in x + n / 2, thus
            # has size 2 * n
            y_ind = y_ind - self.tile_size[0] / 4
            x_ind = x_ind - self.tile_size[1] / 4

        return np.array(list(itertools.product(y_ind, x_ind))).\
            reshape(self.tiles_count + (2,))

    def create_tiles(self, cplx=False):
        """Create a numpy 4D array which will hold the tiles. If *cplx*
        is True, the tiles in the array will be complex-valued.
        """
        ar_type = cfg.NP_CPLX if cplx else cfg.NP_FLOAT

        return np.empty(self.tiles_count + self.tile_size, dtype=ar_type)

    def reconstruct(self, tiles):
        """Reconstruct the whole image from image *tiles*."""
        result = np.empty(self.size, dtype=tiles.dtype)

        if self.outlier:
            # Cut the outlier regions to n / 4 .. 3 * n / 4
            start = self.tile_size[0] / 4, self.tile_size[1] / 4
            end = 3 * start[0], 3 * start[1]
        else:
            start = 0, 0
            end = self.tile_size

        for j in range(tiles.shape[0]):
            for i in range(tiles.shape[1]):
                # If outlier was used, we need to add tile_size / 4 to the
                # index.
                result[self.tile_indices[j, i][0] + start[0]:
                       self.tile_indices[j, i][0] +
                       self.tile_size[0] - start[0],
                       self.tile_indices[j, i][1] + start[1]:
                       self.tile_indices[j, i][1] +
                       self.tile_size[1] - start[1]] = \
                    tiles[j, i][start[0]:end[0], start[1]:end[1]]

        return result
