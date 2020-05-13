"""Synchrotron radiation imaging experiments base module."""
import numpy as np
import pyopencl.array as cl_array
import quantities as q
import logging
import syris.config as cfg
import syris.math as smath
import syris.imageprocessing as ip
from syris.physics import propagate


LOG = logging.getLogger(__name__)


class Experiment(object):

    """A virtual synchrotron experiment base class."""

    def __init__(self, samples, source, detector, propagation_distance, energies):
        self.source = source
        self.samples = samples
        self.detector = detector
        self.propagation_distance = propagation_distance
        self.energies = energies
        self._time = None

    @property
    def time(self):
        """Total time of all samples."""
        if self._time is None:
            self._time = max(
                [obj.trajectory.time for obj in self.samples if obj.trajectory is not None]
            )

        return self._time

    def get_next_time(self, t, pixel_size):
        """Get next time from *t* for all the samples."""
        return min([obj.get_next_time(t, pixel_size) for obj in self.samples])

    def make_source_blur(self, shape, pixel_size, queue=None, block=False):
        """Make geometrical source blurring kernel with *shape* (y, x) size and *pixel_size*. Use
        OpenCL command *queue* and *block* if True.
        """
        d_sample = self.source.sample_distance
        size = self.source.size
        width = (self.propagation_distance * size[1] // d_sample).simplified.magnitude
        height = (self.propagation_distance * size[0] // d_sample).simplified.magnitude
        sigma = (smath.fwnm_to_sigma(height, n=2), smath.fwnm_to_sigma(width, n=2)) * q.m

        return ip.get_gauss_2d(
            shape, sigma, pixel_size=pixel_size, fourier=True, queue=queue, block=block
        )

    def compute_intensity(self, t_0, t_1, shape, pixel_size, queue=None, block=False):
        """Compute intensity between times *t_0* and *t_1*."""
        exp_time = (t_1 - t_0).simplified.magnitude
        image = (
            propagate(
                self.samples,
                shape,
                self.energies,
                self.propagation_distance,
                pixel_size,
                detector=self.detector,
                t=t_0,
            )
            * exp_time
        )

        return image

    def make_sequence(
        self,
        t_start,
        t_end,
        shape=None,
        shot_noise=True,
        amplifier_noise=True,
        source_blur=True,
        queue=None,
    ):
        """Make images between times *t_start* and *t_end*."""
        if queue is None:
            queue = cfg.OPENCL.queue
        shape_0 = self.detector.camera.shape
        if shape is None:
            shape = shape_0
        ps_0 = self.detector.pixel_size
        ps = shape_0[0] / float(shape[0]) * ps_0
        fps = self.detector.camera.fps
        frame_time = 1 / fps
        times = (
            np.arange(
                t_start.simplified.magnitude,
                t_end.simplified.magnitude,
                frame_time.simplified.magnitude,
            )
            * q.s
        )
        image = cl_array.Array(queue, shape, dtype=cfg.PRECISION.np_float)
        source_blur_kernel = None
        if source_blur:
            source_blur_kernel = self.make_source_blur(shape, ps, queue=queue, block=False)

        fmt = "Making sequence with shape {} and pixel size {} from {} to {}"
        LOG.debug(fmt.format(shape, ps, t_start, t_end))

        for t_0 in times:
            image.fill(0)
            t = t_0
            t_next = self.get_next_time(t, ps)
            while t_next < t_0 + frame_time:
                LOG.debug("Motion blur: {} -> {}".format(t, t_next))
                image += self.compute_intensity(t, t_next, shape, ps)
                t = t_next
                t_next = self.get_next_time(t, ps)
            image += self.compute_intensity(t, t_0 + frame_time, shape, ps)
            if source_blur:
                image = ip.ifft_2(ip.fft_2(image) * source_blur_kernel).real
            camera_image = self.detector.camera.get_image(
                image, shot_noise=shot_noise, amplifier_noise=amplifier_noise
            )
            LOG.debug("Image: {} -> {}".format(t_0, t_0 + frame_time))
            yield camera_image
