"""Synchrotron radiation imaging experiments base module."""
import numpy as np
import pyopencl.array as cl_array
import quantities as q
import logging
import syris.config as cfg
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
            self._time = max([obj.trajectory.time for obj in self.samples
                              if obj.trajectory is not None])

        return self._time

    def get_next_time(self, t):
        """Get next time from *t* for all the samples."""
        return min([obj.get_next_time(t, self.detector.pixel_size) for obj in self.samples])

    def compute_intensity(self, t_0, t_1):
        """Compute intensity between times *t_0* and *t_1*."""
        exp_time = (t_1 - t_0).simplified.magnitude
        image = propagate(self.samples, self.detector.camera.shape, self.energies,
                          self.propagation_distance, self.detector.pixel_size,
                          detector=self.detector, t=t_0) * exp_time

        return image

    def make_sequence(self, t_start, t_end, queue=None):
        """Make images between times *t_start* and *t_end*."""
        if queue is None:
            queue = cfg.OPENCL.queue
        fps = self.detector.camera.fps
        frame_time = 1 / fps
        times = np.arange(t_start.simplified.magnitude, t_end.simplified.magnitude,
                          frame_time.simplified.magnitude) * q.s

        image = cl_array.Array(queue, self.detector.camera.shape, dtype=cfg.PRECISION.np_float)

        for t_0 in times:
            image.fill(0)
            t = t_0
            t_next = self.get_next_time(t)
            while t_next < t_0 + frame_time:
                LOG.debug('Motion blur: {} -> {}'.format(t, t_next))
                image += self.compute_intensity(t, t_next)
                t = t_next
                t_next = self.get_next_time(t)
            image += self.compute_intensity(t, t_0 + frame_time)
            camera_image = self.detector.camera.get_image(image)
            LOG.debug('Image: {} -> {}'.format(t_0, t_0 + frame_time))
            yield camera_image
