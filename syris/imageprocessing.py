"""Module for GPU-based image processing."""
import itertools
import numpy as np
import pyopencl as cl
from pyopencl.array import vec
from syris import config as cfg
from syris.gpu import util as g_util


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


def get_gauss_2_f(shape, sigma, pixel_size):
    """Get 2D Gaussian of *shape* (y, x) in Fourier Space
    with standard deviation *sigma* specified as (y, x) and *pixel_size*.
    """
    mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE,
                    size=shape[0] * shape[1] * cfg.PRECISION.cl_cplx)

    cfg.OPENCL.programs['improc'].gauss_2_f(cfg.OPENCL.queue,
                                            shape,
                                            None,
                                            mem,
                                            g_util.make_vfloat2(sigma[1].simplified,
                                                                sigma[0].simplified),
                                            cfg.PRECISION.np_float(pixel_size.simplified))

    return mem


def sum(orig_shape, summed_shape, mem, region, offset,
        average=False, out_mem=None):
    """Sum 2D OpenCL buffer *mem* with *orig_shape* as (y, x). One
    resulting pixel is summed over *region* (y, x) of pixels
    in the original buffer. The resulting buffer has shape *summer_shape*
    (y, x). *Offset* (y, x) is the offset to the original buffer *mem*.
    If *average* is True, the summed pixel is normalized by the
    region area. If *out_mem* is not None, the output buffer
    will be *out_mem*.
    """
    if out_mem is None:
        bpp = mem.size / (orig_shape[0] * orig_shape[1])
        out_mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE,
                            size=summed_shape[0] * summed_shape[1] * bpp)

    cfg.OPENCL.programs['improc'].sum(cfg.OPENCL.queue,
                                      (summed_shape[::-1]),
                                      None,
                                      out_mem,
                                      mem,
                                      vec.make_int2(*region[::-1]),
                                      np.int32(orig_shape[1]),
                                      vec.make_int2(*offset[::-1]),
                                      np.int32(average))

    return out_mem


def _check_tiling(shape, tiles_count):
    """Check if tiling with tile counts *tile_counts* as (y, x) is possible
    for *shape* (y, x).
    """
    if shape[0] % tiles_count[0] != 0 or shape[1] % tiles_count[1] != 0:
        raise ValueError("shape must be a multiple of tile shape.")


class Tiler(object):

    """Class for breaking images into smaller tiles."""

    def __init__(self, shape, tiles_count, outlier=True, supersampling=1,
                 cplx=False):
        """
        Create image tiler for a region of *shape* (y, x) to tiles with (y, x)
        *tiles_count*. If *outlier* is True we want to include outlier regions
        in the tiles, thus they are twice as large (this is used for dealing
        with FFT outlier artifacts). *Supersampling* determines
        the coeffiecient by which the resulting image dimensions will be
        multiplied. If *cplx* is True, the resulting overall image will
        be complex.
        """
        _check_tiling(shape, tiles_count)

        self.tiles_count = tiles_count
        self._outlier_coeff = 2 if outlier else 1
        self.supersampling = supersampling
        self.shape = (shape[0] * self.supersampling,
                      shape[1] * self.supersampling)

        ar_type = cfg.PRECISION.np_cplx if cplx else cfg.PRECISION.np_float

        self._overall = np.empty((self.shape[0] / self.supersampling,
                                  self.shape[1] / self.supersampling),
                                 dtype=ar_type)

    @property
    def result_tile_shape(self):
        """Result tile shape without outlier and supersampling."""
        return tuple([dim / self.supersampling / self._outlier_coeff
                      for dim in self.tile_shape])

    @property
    def outlier(self):
        return bool(self._outlier_coeff - 1)

    @property
    def overall_image(self):
        return self._overall

    @property
    def tile_shape(self):
        """Get the supersampled tile shape based on tile counts
        *tile_counts* as (y, x) and *shape* (y, x).
        """
        return self._outlier_coeff * self.shape[0] / self.tiles_count[0], \
            self._outlier_coeff * self.shape[1] / self.tiles_count[1]

    @property
    def tile_indices(self):
        """Get the supersampled tile indices which are starting points
        of a given tile in (y, x) fashion.
        """
        y_ind = np.array([i * self.tile_shape[0] / self._outlier_coeff
                          for i in range(self.tiles_count[0])])
        x_ind = np.array([i * self.tile_shape[1] / self._outlier_coeff
                          for i in range(self.tiles_count[1])])

        if self.outlier:
            # If the tile starts at x and has a shape n, then with outlier
            # treatment it starts at x - n / 2 and ends in x + n / 2, thus
            # has shape 2 * n
            y_ind = y_ind - self.tile_shape[0] / 4
            x_ind = x_ind - self.tile_shape[1] / 4

        return np.array(list(itertools.product(y_ind, x_ind))).\
            reshape(self.tiles_count + (2,))

    def average(self, tile_mem, out_mem=None):
        """Average OpenCL buffer *tile_mem* based on supersampling
        and outlier specified for the tiler. If *out_mem* is not None,
        it will be used for returning the sum.
        """
        summed_shape = self.result_tile_shape
        offset = [(self._outlier_coeff - 1) * dim / 4
                  for dim in self.tile_shape]

        return sum(self.tile_shape, summed_shape, tile_mem,
                   (self.supersampling, self.supersampling),
                   offset, average=True, out_mem=out_mem)

    def insert(self, tile, indices):
        """Insert a non-supersampled, outlier-free *tile* into the overall
        image. *indices* (y, x) are tile indices in the overall image.
        """
        # Get rid of supersampling and outlier.
        tile_shape = self.result_tile_shape

        self._overall[indices[0] * tile_shape[0]:
                      tile_shape[0] * (indices[0] + 1),
                      indices[1] * tile_shape[1]:
                      tile_shape[1] * (indices[1] + 1)] = tile
