"""Module for GPU-based image processing."""
import glob
import itertools
import logging
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.array import vec
from reikna.cluda import ocl_api
from reikna.fft import FFT
from syris import config as cfg
from syris.gpu import util as g_util
from syris.math import fwnm_to_sigma
from syris.util import get_magnitude, make_tuple, next_power_of_two, read_image, save_image


LOG = logging.getLogger(__name__)


def fft_2(data, queue=None, block=True):
    """2D FFT executed on *data*. *block* specifies if the execution will wait until the scheduled
    FFT kernels finish. The transformation is done in-place if *data* is a pyopencl Array class and
    has complex data type, otherwise the data is converted first.
    """
    return _fft_2(data, inverse=False, queue=queue, block=block)


def ifft_2(data, queue=None, block=True):
    """2D inverse FFT executed on *data*. *block* specifies if the execution will wait until the
    scheduled FFT kernels finish. The transformation is done in-place if *data* is a pyopencl Array
    class and has complex data type, otherwise the data is converted first.  """
    return _fft_2(data, inverse=True, queue=queue, block=block)


def _fft_2(data, inverse=False, queue=None, block=True):
    """Execute FFT on *data*, which is first converted to a pyopencl array and retyped to
    complex.
    """
    if not queue:
        queue = cfg.OPENCL.queue
    thread = ocl_api().Thread(queue)
    data = g_util.get_array(data, queue=queue)
    if data.dtype != cfg.PRECISION.np_cplx:
        data = data.astype(cfg.PRECISION.np_cplx)

    if queue not in cfg.OPENCL.fft_plans:
        cfg.OPENCL.fft_plans[queue] = {}
    if data.shape not in cfg.OPENCL.fft_plans[queue]:
        LOG.debug('Creating FFT Plan for {} and shape {}'.format(queue, data.shape))
        _fft = FFT(data, axes=(0, 1))
        cfg.OPENCL.fft_plans[queue][data.shape] = _fft.compile(thread, fast_math=False)
    plan = cfg.OPENCL.fft_plans[queue][data.shape]

    LOG.debug('fft_2, shape: %s, inverse: %s', data.shape, inverse)
    # plan.execute(data.data, inverse=inverse, wait_for_finish=block)
    plan(data, data, inverse=inverse)
    if block:
        thread.synchronize()

    return data


def get_gauss_2d(shape, sigma, pixel_size=1, fourier=False, queue=None, block=False):
    """Get 2D Gaussian of *shape* with standard deviation *sigma* and *pixel_size*. If *fourier* is
    True the fourier transform of it is returned so it is faster for usage by convolution. Use
    command *queue* if specified. If *block* is True, wait for the kernel to finish.
    """
    shape = make_tuple(shape)
    pixel_size = get_magnitude(make_tuple(pixel_size))
    sigma = get_magnitude(make_tuple(sigma))
    LOG.debug('get_gauss_2d, shape: %s, sigma: %s, pixel size: %s, fourier: %s',
              shape, sigma, pixel_size, fourier)

    if queue is None:
        queue = cfg.OPENCL.queue
    out = cl.array.Array(queue, shape, dtype=cfg.PRECISION.np_float)

    if fourier:
        ev = cfg.OPENCL.programs['improc'].gauss_2d_f(queue,
                                                      shape[::-1],
                                                      None,
                                                      out.data,
                                                      g_util.make_vfloat2(sigma[1], sigma[0]),
                                                      g_util.make_vfloat2(pixel_size[1],
                                                      pixel_size[0]))
    else:
        ev = cfg.OPENCL.programs['improc'].gauss_2d(queue,
                                                    shape[::-1],
                                                    None,
                                                    out.data,
                                                    g_util.make_vfloat2(sigma[1], sigma[0]),
                                                    g_util.make_vfloat2(pixel_size[1],
                                                    pixel_size[0]))
    if block:
        ev.wait()

    return out


def pad(image, region=None, out=None, value=0, queue=None, block=False):
    """Pad a 2D *image*. *region* is the region to pad as (y_0, x_0, height, width). If not
    specified, the next power of two dimensions are used and the image is centered in the padded
    one. The final image dimensions are height x width and the filling starts at (y_0, x_0), *out*
    is the pyopencl Array instance, if not specified it will be created. *out* is also returned.
    *value* is the padded value. If *block* is True, wait for the copy to finish.
    """
    if region is None:
        shape = tuple([next_power_of_two(n) for n in image.shape])
        y_0 = (shape[0] - image.shape[0]) / 2
        x_0 = (shape[1] - image.shape[1]) / 2
        region = (y_0, x_0) + shape
    if queue is None:
        queue = cfg.OPENCL.queue
    if out is None:
        out = cl_array.zeros(queue, (region[2], region[3]), dtype=image.dtype) + value
    image = g_util.get_array(image, queue=queue)

    n_bytes = image.dtype.itemsize
    y_0, x_0, height, width = region
    src_origin = (0, 0, 0)
    dst_origin = (n_bytes * x_0, y_0, 0)
    region = (n_bytes * image.shape[1], image.shape[0], 1)
    LOG.debug('pad, shape: %s, src_origin: %s, dst_origin: %s, region: %s', image.shape,
              src_origin, dst_origin, region)

    _copy_rect(image, out, src_origin, dst_origin, region, queue, block=block)

    return out


def crop(image, region, out=None, queue=None, block=False):
    """Crop a 2D *image*. *region* is the region to crop as (y_0, x_0, height, width), *out* is the
    pyopencl Array instance, if not specified it will be created. *out* is also returned. If *block*
    is True, wait for the copy to finish.
    """
    if queue is None:
        queue = cfg.OPENCL.queue
    if out is None:
        out = cl.array.Array(queue, (region[2], region[3]), dtype=image.dtype)
    image = g_util.get_array(image, queue=queue)

    n_bytes = image.dtype.itemsize
    y_0, x_0, height, width = region
    src_origin = (n_bytes * x_0, y_0, 0)
    dst_origin = (0, 0, 0)
    region = (n_bytes * width, height, 1)
    LOG.debug('crop, shape: %s, src_origin: %s, dst_origin: %s, region: %s', image.shape,
              src_origin, dst_origin, region)

    _copy_rect(image, out, src_origin, dst_origin, region, queue, block=block)

    return out


def bin_image(image, summed_shape, offset=(0, 0), average=False, out=None, queue=None,
              block=False):
    """Bin an *image*. The resulting buffer has shape *summed_shape* (y, x). *Offset* (y, x) is the
    offset to the original *image*. *summed_shape* has to be a divisor of the original shape minus
    the *offset*. If *average* is True, the summed pixel is normalized by the region area.  *out* is
    the pyopencl Array instance, if not specified it will be created. *out* is also returned. If
    *block* is True, wait for the copy to finish.
    """
    if queue is None:
        queue = cfg.OPENCL.queue
    if out is None:
        out = cl.array.Array(queue, summed_shape, dtype=cfg.PRECISION.np_float)
    image = g_util.get_array(image, queue=queue)
    orig_shape = (image.shape[0] - offset[0], image.shape[1] - offset[1])
    region = (orig_shape[0] / summed_shape[0], orig_shape[1] / summed_shape[1])
    if orig_shape[0] % summed_shape[0] or orig_shape[1] % summed_shape[1]:
        raise RuntimeError('Final shape {} must be a divisor '.format(summed_shape) +
                           'of the original shape {}'.format(image.shape))
    LOG.debug('bin_image, shape: %s, summed_shape: %s, offset: %s, average: %s', image.shape,
              summed_shape, offset, average)

    ev = cfg.OPENCL.programs['improc'].sum(queue,
                                           (summed_shape[::-1]),
                                           None,
                                           out.data,
                                           image.data,
                                           vec.make_int2(*region[::-1]),
                                           np.int32(image.shape[1]),
                                           vec.make_int2(*offset[::-1]),
                                           np.int32(average))
    if block:
        ev.wait()

    return out


def decimate(image, shape, sigma=None, average=False, queue=None, block=False):
    """Decimate *image* so that its dimensions match the final *shape*, which has to be a divisor of
    the original shape. Remove low frequencies by a Gaussian filter with *sigma* pixels. If *sigma*
    is None, use the FWHM of one low resolution pixel. Use command *queue*, if *block* is True, wait
    for the copy to finish.
    """
    if queue is None:
        queue = cfg.OPENCL.queue
    image = g_util.get_array(image, queue=queue)
    shape = make_tuple(shape)
    pow_shape = tuple([next_power_of_two(n) for n in image.shape])
    orig_shape = image.shape
    if image.shape != pow_shape:
        image = pad(image, region=(0, 0) + pow_shape, queue=queue)
    if sigma is None:
        sigma = tuple([fwnm_to_sigma(float(image.shape[i]) / shape[i], n=2) for i in range(2)])

    LOG.debug('decimate, shape: %s, final_shape: %s, sigma: %s, average: %s', image.shape, shape,
              sigma, average)

    fltr = get_gauss_2d(image.shape, sigma, fourier=True, queue=queue, block=block)
    image = image.astype(cfg.PRECISION.np_cplx)
    fft_2(image, queue=queue, block=block)
    image *= fltr
    ifft_2(image, queue=queue, block=block)
    image = crop(image.real, (0, 0) + orig_shape, queue=queue, block=block)

    return bin_image(image, shape, average=average, queue=queue, block=block)


def blur_with_gaussian(image, sigma, queue=None, block=False):
    """Blur *image* with a gaussian kernel, where *sigma* is the standard deviation. Use command
    *queue*, if *block* is True, wait for the copy to finish.
    """
    fltr = get_gauss_2d(image.shape, sigma, fourier=True, queue=queue, block=block)
    image = image.astype(cfg.PRECISION.np_cplx)
    image = fft_2(image, queue=queue, block=block)
    image *= fltr
    ifft_2(image, queue=queue, block=block)

    return image.real


def rescale(image, shape, sampler=None, queue=None, out=None, block=False):
    """Rescale *image* to *shape* and use *sampler* which is a :class:`pyopencl.Sampler` instance.
    Use OpenCL *queue* and *out* pyopencl Array. If *block* is True, wait for the copy to finish.
    """
    if cfg.PRECISION.cl_float == 8:
        raise TypeError('Double precision mode not supported')
    shape = make_tuple(shape)
    # OpenCL order
    factor = float(shape[1]) / image.shape[1], float(shape[0]) / image.shape[0]
    LOG.debug('rescale, shape: %s, final_shape: %s, factor: %s', image.shape, shape, factor)

    if queue is None:
        queue = cfg.OPENCL.queue
    if out is None:
        out = cl.array.Array(queue, shape, dtype=cfg.PRECISION.np_float)

    if not sampler:
        sampler = cl.Sampler(cfg.OPENCL.ctx, False, cl.addressing_mode.CLAMP_TO_EDGE, cl.filter_mode.LINEAR)
    image = g_util.get_image(image)

    ev = cfg.OPENCL.programs['improc'].rescale(queue,
                                               shape[::-1],
                                               None,
                                               image,
                                               out.data,
                                               sampler,
                                               g_util.make_vfloat2(*factor))
    if block:
        ev.wait()

    return out


def compute_intensity(wavefield, queue=None, out=None, block=False):
    if queue is None:
        queue = cfg.OPENCL.queue
    if out is None:
        out = cl.array.Array(queue, wavefield.shape, dtype=cfg.PRECISION.np_float)
    wavefield = g_util.get_array(wavefield, queue=queue)

    ev = cfg.OPENCL.programs['improc'].compute_intensity(queue,
                                                         wavefield.shape[::-1],
                                                         None,
                                                         wavefield.data,
                                                         out.data)
    if block:
        ev.wait()

    return out


def varconvolve(kernel_name, shape, kernel_args, local_size=None, program=None, queue=None,
                block=False):
    """Variable convolution with OpenCL kernel function *kernel_name*, gloal size *shape* (y, x),
    kernel arguments *kernel_args*, work group size *local_size* (can be None, i.e. OpenCL will
    determine it automatically), OpenCL *program* (can be None in which case the default syris
    variable convolution program is used with all predefined kernels). *queue* is the command queue,
    if *block* is True wait for the kernel to finish. Return OpenCL event from the kernel execution.

    .. math::

        (f \\ast g)(x, y) = \\int_{-\\infty}^{\\infty} \\int_{-\\infty}^{\\infty} \\
                f(x, y, \\xi, \\eta) g(x - \\xi, y - \\eta) d\\xi d\\eta

    """
    if not program:
        program = cfg.OPENCL.programs['varconv']
    if queue is None:
        queue = cfg.OPENCL.queue
    LOG.debug('varconvolve, shape: %s, kernel: %s', shape, kernel_name)

    ev = getattr(program, kernel_name)(queue, shape[::-1], local_size, *kernel_args)

    if block:
        ev.wait()

    return ev


def _varconvolve_2d_parametrized(image, parameters, kernel_name, sampler=None, queue=None,
                                 out=None, block=False):
    """Variable convolution of *image* with *parameters*, use OpoenCL kernel *kernel_name*,
    *sampler*, *queue*, *out* and wait if *block* is True. Return *out*.
    """
    if queue is None:
        queue = cfg.OPENCL.queue
    if out is None:
        out = cl.array.Array(queue, image.shape, dtype=cfg.PRECISION.np_float)
    if sampler is None:
        sampler = cl.Sampler(queue.context, False, cl.addressing_mode.CLAMP_TO_EDGE,
                             cl.filter_mode.NEAREST)
    if not isinstance(parameters, cl_array.Array):
        params_host = np.empty(parameters[0].shape, dtype=cfg.PRECISION.vfloat2)
        params_host['y'] = g_util.get_host(parameters[0])
        params_host['x'] = g_util.get_host(parameters[1])
        parameters = cl_array.to_device(queue, params_host)
    if parameters.shape != image.shape:
        raise ValueError("Parameters shape '{}' differs from image shape '{}'".
                         format(parameters.shape, image.shape))
    image = g_util.get_image(image, queue=queue)
    args = (image, out.data, sampler, cl_array.vec.make_int2(0, 0), parameters.data)

    varconvolve(kernel_name, image.shape[::-1], args, queue=queue, block=block)

    return out


def varconvolve_gauss(image, sigmas, normalized=True, sampler=None, queue=None,
                      out=None, block=False):
    """Variable convolution of input *image* with a Gaussian with y and x sigmas. *sigmas* specify
    the convolution kernel y and x sigmas for every output point. They are specified as two 2D
    arrays and can be either a tuple of two 2D arrays or a pyopencl.array.Array instance with
    vfloat2 data type, meaning both 2D arrays are encoded in it. If *normalized* is True the
    convolution kernel sum is always 1. Use OpenCL *sampler*, command *queue*, *out* as output and
    wait for execution end if *block* is True.
    Convolution window is always odd-shaped and the middle pixel is set to 0. This means that if the
    *sigmas* are smaller numbers than 1, the convolution returns the original image.
    """
    kernel_name = 'varconvolve_gauss'
    if normalized:
        kernel_name += '_normalized'
    return _varconvolve_2d_parametrized(image, sigmas, kernel_name, sampler=sampler,
                                        queue=queue, out=out, block=block)


def varconvolve_disk(image, radii, normalized=True, smooth=True, sampler=None, queue=None,
                     out=None, block=False):
    """Variable convolution of input *image* with an elliptical disk with y and x radii. *radii*
    specify the convolution kernel disk y and x radius for every output point. They are specified as
    two 2D arrays and can be either a tuple of two 2D arrays or a pyopencl.array.Array instance with
    vfloat2 data type, meaning both 2D arrays are encoded in it. If *normalized* is True the
    convolution kernel sum is always 1. Use OpenCL *sampler*, command *queue*, *out* as output and
    wait for execution end if *block* is True.
    Convolution window is always odd-shaped and the middle pixel is set to 0. This means that if the
    *radii* are smaller numbers than 1, the convolution returns the original image. This has a
    consequence that it is not possible to create a disk with even number of pixels accross one of
    the principal axes, so the disk radius will be exact from the middle if you specify it in half
    pixels, e.g. if the radius is 1.5, then pixels [-1, 0, 1] will be selected, i.e. the disk
    diameter is 3 pixels.
    """
    kernel_name = 'varconvolve_disk'
    if smooth:
        kernel_name += '_smooth'
    if normalized:
        kernel_name += '_normalized'
    return _varconvolve_2d_parametrized(image, radii, kernel_name, sampler=sampler,
                                        queue=queue, out=out, block=block)


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
    command queues to use, *args* and *kwargs* are additional arguments passed to *func*. Returns a
    generator.
    """
    if iterable is None:
        iterable = make_tile_offsets(shape, tile_shape, outlier=outlier)
    if kwargs is None:
        kwargs = {}

    if queues is None:
        return (func(item, *args, **kwargs) for item in iterable)
    else:
        # Use multiple comand queues
        return (item for item in g_util.qmap(func, iterable, queues=queues, args=args,
                                             kwargs=kwargs))


def save_tiles(prefix, tiles):
    """Save *tiles* to linearly indexed files formed by *prefix* and the tile number."""
    for i, tile in enumerate(tiles):
        save_image(prefix.format(i), tile)


def read_tiles(prefix):
    """Read tiles from disk using the glob module for pattern expansion. Returns a generator."""
    names = sorted(glob.glob(prefix))

    return (read_image(name) for name in names)


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


def _copy_rect(src, dst, src_origin, dst_origin, region, queue, block=False):
    """Copy a rectangular OpenCL buffer *region* from *src* to *dst*, where both are a pyopencl
    Array instance. *src_origin* and *dst_origin* specify the offsets. *queue* is an OpenCL command
    queue. If *block* is True, wait for the copy to finish.
    """
    n_bytes = src.dtype.itemsize
    src_pitches = (n_bytes * src.shape[1], n_bytes * src.shape[1] * src.shape[0])
    dst_pitches = (n_bytes * dst.shape[1], n_bytes * dst.shape[1] * dst.shape[0])

    ev = cl.enqueue_copy(queue, dst.data, src.data, src_origin=src_origin,
                         dst_origin=dst_origin, region=region,
                         src_pitches=src_pitches, dst_pitches=dst_pitches)
    if block:
        ev.wait()
