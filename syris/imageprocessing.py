"""Module for GPU-based image processing."""
import glob
import itertools
import logging
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.array import vec
from pyfft.cl import Plan
from syris import config as cfg
from syris.gpu import util as g_util
from syris.math import fwnm_to_sigma
from syris.util import get_magnitude, make_tuple, next_power_of_two, read_image, save_image


LOG = logging.getLogger(__name__)


def fft_2(data, plan=None, queue=None, wait_for_finish=True):
    """2D FFT executed on *data* by a *plan*. If *plan* is None, it is created either for *queue* if
    given, otherwise for the default opencl queue. *wait_for_finish* specifies if the execution will
    wait until the scheduled FFT kernels finish. The transformation is done in-place if *data* is a
    pyopencl Array class and has complex data type, otherwise the data is converted first.
    """
    return _fft_2(data, inverse=False, plan=plan, queue=queue, wait_for_finish=wait_for_finish)


def ifft_2(data, plan=None, queue=None, wait_for_finish=True):
    """2D inverse FFT executed on *data* by a *plan*. If *plan* is None, it is created either for
    *queue* if given, otherwise for the default opencl queue. *wait_for_finish* specifies if the
    execution will wait until the scheduled FFT kernels finish. The transformation is done in-place
    if *data* is a pyopencl Array class and has complex data type, otherwise the data is converted
    first.
    """
    return _fft_2(data, inverse=True, plan=plan, queue=queue, wait_for_finish=wait_for_finish)


def _fft_2(data, inverse=False, plan=None, queue=None, wait_for_finish=True):
    """Execute FFT on *data*, which is first converted to a pyopencl array and retyped to
    complex.
    """
    data = g_util.get_array(data)
    if data.dtype != cfg.PRECISION.np_cplx:
        data = data.astype(cfg.PRECISION.np_cplx)

    if not plan:
        if not queue:
            queue = cfg.OPENCL.queue
        if queue not in cfg.OPENCL.fft_plans:
            cfg.OPENCL.fft_plans[queue] = {}
        if data.shape not in cfg.OPENCL.fft_plans[queue]:
            LOG.debug('Creating FFT Plan for {} and shape {}'.format(queue, data.shape))
            cfg.OPENCL.fft_plans[queue][data.shape] = Plan(data.shape, queue=queue)
        plan = cfg.OPENCL.fft_plans[queue][data.shape]

    plan.execute(data.data, inverse=inverse, wait_for_finish=wait_for_finish)

    return data


def get_gauss_2d(shape, sigma, pixel_size=1, fourier=False, queue=None):
    """Get 2D Gaussian of *shape* with standard deviation *sigma* and *pixel_size*. If *fourier* is
    True the fourier transform of it is returned so it is faster for usage by convolution. Use
    command *queue* if specified.
    """
    shape = make_tuple(shape)
    pixel_size = get_magnitude(make_tuple(pixel_size))
    sigma = get_magnitude(make_tuple(sigma))

    if queue is None:
        queue = cfg.OPENCL.queue
    out = cl.array.Array(queue, shape, dtype=cfg.PRECISION.np_float)

    if fourier:
        cfg.OPENCL.programs['improc'].gauss_2d_f(queue,
                                                 shape[::-1],
                                                 None,
                                                 out.data,
                                                 g_util.make_vfloat2(sigma[1], sigma[0]),
                                                 g_util.make_vfloat2(pixel_size[1], pixel_size[0]))
    else:
        cfg.OPENCL.programs['improc'].gauss_2d(queue,
                                               shape[::-1],
                                               None,
                                               out.data,
                                               g_util.make_vfloat2(sigma[1], sigma[0]),
                                               g_util.make_vfloat2(pixel_size[1], pixel_size[0]))

    return out


def pad(image, region=None, out=None, queue=None):
    """Pad a 2D *image*. *region* is the region to pad as (y_0, x_0, height, width). If not
    specified, the next power of two dimensions are used and the image is centered in the padded
    one. The final image dimensions are height x width and the filling starts at (y_0, x_0), *out*
    is the pyopencl Array instance, if not specified it will be created. *out* is also returned.
    """
    if region is None:
        shape = tuple([next_power_of_two(n) for n in image.shape])
        y_0 = (shape[0] - image.shape[0]) / 2
        x_0 = (shape[1] - image.shape[1]) / 2
        region = (y_0, x_0) + shape
    if queue is None:
        queue = cfg.OPENCL.queue
    if out is None:
        out = cl_array.zeros(queue, (region[2], region[3]), dtype=image.dtype)
    image = g_util.get_array(image)

    n_bytes = image.dtype.itemsize
    y_0, x_0, height, width = region
    src_origin = (0, 0, 0)
    dst_origin = (n_bytes * x_0, y_0, 0)
    region = (n_bytes * image.shape[1], image.shape[0], 1)

    _copy_rect(image, out, src_origin, dst_origin, region, queue)

    return out


def crop(image, region, out=None, queue=None):
    """Crop a 2D *image*. *region* is the region to crop as (y_0, x_0, height, width), *out* is the
    pyopencl Array instance, if not specified it will be created. *out* is also returned.
    """
    if queue is None:
        queue = cfg.OPENCL.queue
    if out is None:
        out = cl.array.Array(queue, (region[2], region[3]), dtype=image.dtype)
    image = g_util.get_array(image)

    n_bytes = image.dtype.itemsize
    y_0, x_0, height, width = region
    src_origin = (n_bytes * x_0, y_0, 0)
    dst_origin = (0, 0, 0)
    region = (n_bytes * width, height, 1)

    _copy_rect(image, out, src_origin, dst_origin, region, queue)

    return out


def bin_image(image, summed_shape, offset=(0, 0), average=False, out=None, queue=None):
    """Bin a *image*. The resulting buffer has shape *summer_shape* (y, x).  *Offset* (y, x) is the
    offset to the original *image*.  If *average* is True, the summed pixel is normalized by the
    region area. *out* is the pyopencl Array instance, if not specified it will be created. *out* is
    also returned.
    """
    if queue is None:
        queue = cfg.OPENCL.queue
    if out is None:
        out = cl.array.Array(queue, summed_shape, dtype=cfg.PRECISION.np_float)
    image = g_util.get_array(image)
    region = ((image.shape[0] - offset[0]) / summed_shape[0],
              (image.shape[1] - offset[1]) / summed_shape[1])

    cfg.OPENCL.programs['improc'].sum(queue,
                                      (summed_shape[::-1]),
                                      None,
                                      out.data,
                                      image.data,
                                      vec.make_int2(*region[::-1]),
                                      np.int32(image.shape[1]),
                                      vec.make_int2(*offset[::-1]),
                                      np.int32(average))

    return out


def decimate(image, shape, sigma=None, average=False, queue=None, plan=None):
    """Decimate *image* so that its dimensions match the final *shape*. Remove low frequencies by a
    Gaussian filter with *sigma* pixels. If *sigma* is None, use the FWHM of one low resolution
    pixel. Use command *queue* and FFT *plan* if specified.
    """
    if queue is None:
        queue = cfg.OPENCL.queue
    image = g_util.get_array(image)
    image = image.astype(cfg.PRECISION.np_cplx)
    pow_shape = tuple([next_power_of_two(n) for n in image.shape])
    orig_shape = image.shape
    if image.shape != pow_shape:
        image = pad(image, region=(0, 0) + pow_shape, queue=queue)
    if not plan:
        plan = Plan(image.shape, queue=queue)
    if sigma is None:
        sigma = tuple([fwnm_to_sigma(float(image.shape[i]) / shape[i], n=2) for i in range(2)])

    LOG.debug('Decimating {} -> {} with sigma {}'.format(image.shape, shape, sigma))

    fltr = get_gauss_2d(image.shape, sigma, fourier=True, queue=queue)
    fft_2(image, plan, wait_for_finish=True)
    image *= fltr
    ifft_2(image, plan, wait_for_finish=True)
    image = crop(image, (0, 0) + orig_shape, queue=queue)

    return bin_image(image.real, shape, average=average, queue=queue)


def rescale(image, shape, sampler=None, queue=None, out=None):
    """Rescale *image* to *shape* and use *sampler* which is a :class:`pyopencl.Sampler` instance.
    Use OpenCL *queue* and *out* pyopencl Array.
    """
    if cfg.PRECISION.cl_float == 8:
        raise TypeError('Double precision mode not supported')
    shape = make_tuple(shape)
    # OpenCL order
    factor = float(shape[1]) / image.shape[1], float(shape[0]) / image.shape[0]

    if queue is None:
        queue = cfg.OPENCL.queue
    if out is None:
        out = cl.array.Array(queue, shape, dtype=cfg.PRECISION.np_float)

    if not sampler:
        sampler = cl.Sampler(cfg.OPENCL.ctx, False, cl.addressing_mode.NONE, cl.filter_mode.LINEAR)
    image = g_util.get_image(image)

    cfg.OPENCL.programs['improc'].rescale(queue,
                                          shape[::-1],
                                          None,
                                          image,
                                          out.data,
                                          sampler,
                                          g_util.make_vfloat2(*factor))

    return out


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

    def average(self, tile, out=None):
        """Average :class:`pyopencl.array.Array` *tile* based on supersampling and outlier specified
        for the tiler. If *out* is not None, it will be used for returning the sum.
        """
        summed_shape = self.result_tile_shape
        offset = [(self._outlier_coeff - 1) * dim / 4 for dim in self.tile_shape]

        return bin_image(tile, summed_shape, offset, average=True, out=out)

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


def make_tile_offsets(shape, tile_shape, outlier=(0, 0)):
    """Make tile offsets in pixels so that one tile has *tile_shape* and all tiles form an image of
    *shape*. *outlier* specifies the the overlap of the tiles, so if the outlier width is m, the
    tile overlaps with the previous and next tiles by m / 2. If the tile width is n, the tile must
    be cropped to (m / 2, n - n / 2) before it can be placed into the resulting image. This is
    convenient for convolution outlier treatment.
    """
    y_starts = np.arange(0, shape[0], tile_shape[0] - outlier[0]) - outlier[0] / 2
    x_starts = np.arange(0, shape[1], tile_shape[1] - outlier[1]) - outlier[1] / 2

    return list(itertools.product(y_starts, x_starts))


def make_tiles(func, shape, tile_shape, iterable=None, outlier=(0, 0), queues=None,
               args=(), kwargs=None):
    """Make tiles using *func* which can either have signature func(item, *args, **kwargs) or
    func(item, queue, *args, **kwargs), where queue is the OpenCL command queue. In the latter case,
    multiple command queues are mapped to different computation items. *shape* (y, x) is the final
    image shape, *tile_shape* (y, x) is the shape of one tile, *iterable* is the sequence to be
    mapped to *func*, if not specified, the offsets from :func:`.make_tile_offsets` are used.
    *outlier* (y, x) is the amount of overlapping region between tiles, *queues* are the OpenCL
    command queues to use, *args* and *kwargs* are additional arguments passed to *func*.
    """
    if iterable is None:
        iterable = make_tile_offsets(shape, tile_shape, outlier=outlier)
    if kwargs is None:
        kwargs = {}

    if queues is None:
        for item in iterable:
            yield func(item, *args, **kwargs)
    else:
        # Use multiple comand queues
        for item in g_util.qmap(func, iterable, queues=queues, args=args, kwargs=kwargs):
            yield item


def save_tiles(prefix, tiles):
    """Save *tiles* to linearly indexed files formed by *prefix* and the tile number."""
    for i, tile in enumerate(tiles):
        save_image(prefix.format(i), tile)


def read_tiles(prefix):
    """Read tiles from disk using the glob module for pattern expansion."""
    names = sorted(glob.glob(prefix))

    for name in names:
        yield read_image(name)


def merge_tiles(tiles, num_tiles=None, outlier=(0, 0)):
    """Merge *tiles* which is a list to one large image. *num_tiles* is a tuple specifying the
    number of tiles as (y, x) or None, meaning there is equal number of tiles in both dimensions.
    The tiles must be stored in the row-major order.
    """
    n, m = get_num_tiles(tiles, num_tiles=num_tiles)
    tile_shape = tiles[0].shape
    crop_shape = (tile_shape[0] - outlier[0], tile_shape[1] - outlier[1])
    result = np.zeros((n * crop_shape[0], m * crop_shape[1]), dtype=tiles[0].dtype)

    for j in range(n):
        for i in range(m):
            tile = g_util.get_host(tiles[j * m + i])[outlier[0] / 2:tile_shape[0] - outlier[0] / 2,
                                                     outlier[1] / 2:tile_shape[1] - outlier[1] / 2]
            result[j * crop_shape[0]:(j + 1) * crop_shape[0],
                   i * crop_shape[1]:(i + 1) * crop_shape[1]] = tile

    return result


def get_num_tiles(tiles, num_tiles=None):
    """Determine number of tiles in the *tiles* list."""
    if num_tiles is None:
        num_tiles = int(np.sqrt(len(tiles)))
        if num_tiles ** 2 != len(tiles):
            raise ValueError('There must be equal number of tiles in both dimensions if '
                             'num_tiles is not specified')
        num_tiles = (num_tiles, num_tiles)

    return num_tiles


def _copy_rect(src, dst, src_origin, dst_origin, region, queue):
    """Copy a rectangular OpenCL buffer *region* from *src* to *dst*, where both are a pyopencl
    Array instance. *src_origin* and *dst_origin* specify the offsets. *queue* is an OpenCL command
    queue.
    """
    n_bytes = src.dtype.itemsize
    src_pitches = (n_bytes * src.shape[1], n_bytes * src.shape[1] * src.shape[0])
    dst_pitches = (n_bytes * dst.shape[1], n_bytes * dst.shape[1] * dst.shape[0])

    cl.enqueue_copy_buffer_rect(queue, src.data, dst.data, src_origin, dst_origin, region,
                                src_pitches, dst_pitches)
