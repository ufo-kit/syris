"""Synchrotron radiation imaging experiments base module."""
import quantities as q
import logging
import math


LOGGER = logging.getLogger(__name__)


class Experiment(object):

    """A virtual synchrotron experiment base class."""

    def __init__(self, sample, source, scintillator, detector, tiler,
                 filters=None, spatial_incoherence=True):
        """
        """
        self.source = source
        self.filters = filters
        self.sample = sample
        self.scintillator = scintillator
        self.detector = detector
        self.tiler = tiler
        self.spatial_incoherence = spatial_incoherence
        self._time = None

    @property
    def super_pixel_size(self):
        """Supersampled pixel size."""
        return self.detector.pixel_size / self.tiler.supersampling

    @property
    def time(self):
        if self._time is None:
            self._time = max([gr_obj.trajectory.time
                              for gr_obj in self.sample.objects])
        return self._time

    def _init_thickness_cache(self):
        thicknesses = {}
        for material in self.sample.materials:
            thicknesses[material] = {}

        return thicknesses

    def make_x_ray_image(self, abs_time):
        r"""
        Make an X-ray image of a sample at a given *abs_time*. The result
        is a superposition of projections at all source energies.
        An X-ray image is calculated as follows

        .. math::
            :nowrap:

            \begin{eqnarray}
                T(\vec{x},\lambda) & = & e^{-\frac{2 \pi}{\lambda}\sum_i
                p_i(\vec{x}) \left [ \beta_i + i\delta_i\right ]} \\
                u(\vec{x}, 0, \lambda) & = & u_{inc}(\vec{x}, 0, \lambda)
                \cdot T(\vec{x},\lambda) \\
                u(\vec{x},d,\lambda) & = & \frac{e^{ikd}}{id\lambda}
                \int u(\vec{\eta}, 0, \lambda) \cdot e^{i \frac{\pi}
                {d \lambda}(\vec{x} - \vec{\eta})^2} d \vec{\eta} \\
                I(\vec{x}, d, t) & = & \sum_\lambda{|u(\vec{x},d,\lambda)|
                ^ 2}.
            \end{eqnarray}

        :math:`T(\vec{x},\lambda)` is the sample transfer function,
        :math:`p` is projected thickness for one material type,
        :math:`\delta` and :math:`\beta` form the refractive index,
        :math:`u` is wavefield, whereas :math:`u_{inc}` is the
        incident wavefield, :math:`d` is the propagation distance,
        :math:`I(\vec{x}, d, t)` are intensities and :math:`\vec{x}`
        and :math:`\vec{\eta}` are spatial coordinates on the detector
        and sample, respectively.

        """
        for energy in self.source.energies:
            for material in self.sample.materials:
                # Add p_i * r_i of each material.
                pass
            # calculate u_e = u_0 * e ^ (\sum_i{r_i * p_i})
            pass
            # propagate
            pass
            # transform to intensities, I = |U| ^ 2
            pass
            # add to result
            pass

    def make_visible_light_image(self, x_ray_image):
        """
        Apply visible light path on an *x_ray_image*. It is
        not necessary to apply spatial coherence and detector PSF
        all the time thanks to convolution distributivity. Not
        only superposition of all energies, but also superposition
        of all sample subpositions are used to calculate one visible
        light image.
        """
        # PSF + spatial coherence
        pass
        # attenuate by vis. light optical elements
        pass
        # add noise
        pass

    def run(self):
        """Conduct the experiment."""
        d_t = self.detector.camera.exp_time
        cur_t = 0 * q.s
        frame = 1
        num_frames = int(math.ceil(self.time / d_t))

        image = None

        while frame < num_frames:
            # Recalculate graphical objects which moved.
            next_t = self.sample.move(cur_t)

            # While for the case objects are slower than the frame rate
            # and we need to record more frames before some object moves.
            while frame * d_t <= next_t:
                # yield a visible light image
                pass
                LOGGER.debug("Writing out image {0}".format(frame))
                frame += 1

            duration = next_t - cur_t
            LOGGER.debug("Calculating image {0} in times {1} - {2}".
                         format(frame, cur_t, next_t))
            # superimpose subpositions until the exposure time is over
            pass

            cur_t = next_t

    def create_geometry(self):
        pass

    def propagate(self):
        pass

    def detect(self):
        pass
