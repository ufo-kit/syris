"""Synchrotron radiation imaging experiments module. You will find
radiography as well as tomography experiment classes here for conducting
virtual experiments.
"""


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

#     def conduct(self):
#         """Conduct the experiment."""
#         d_t = self.detector.camera.exp_time
#         times = np.arange(0, self.time + d_t, d_t)
#         thicknesses = self._init_thickness_cache()
#
#         for time_i in times:
#             sub_times = []
# First cache the thicknesses for all materials.
#             for subtime_i in range(len(sub_times)):
#
#                 for material in self.sample.materials:
#                     thicknesses[material][subtime_i]
#
#             thicknesses = self._init_thickness_cache()

    def create_geometry(self):
        pass

    def propagate(self):
        pass

    def detect(self):
        pass
