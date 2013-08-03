"""Synchrotron radiation imaging experiments module. You will find
radiography as well as tomography experiment classes here for conducting
virtual experiments.
"""


class Experiment(object):

    """A virtual synchrotron experiment base class."""

    def __init__(self, source, scintillator, detector, tiler, filters=[],
                 samples=[], spatial_incoherence=True):
        """
        """
        if spatial_incoherence and len(samples) > 1:
            raise ValueError("Spatial incoherence cannot be modeled for " +
                             "experiments with more than one sample.")
        self.source = source
        self.filters = filters
        self._samples = samples
        self.scintillator = scintillator
        self.detector = detector
        self.tiler = tiler
        self.spatial_incoherence = spatial_incoherence

    @property
    def super_pixel_size(self):
        """Supersampled pixel size."""
        return self.detector.pixel_size / self.tiler.supersampling

    @property
    def samples(self):
        return tuple(self._samples)

    def add_sample(self, sample, propagation_distance):
        """Add *sample* to experiment samples and set the
        *propagation_distance* of the wavefield behind it.
        """
        if self.spatial_incoherence and len(self._samples) > 0:
            raise ValueError("Spatial incoherence cannot be modeled for " +
                             "experiments with more than one sample.")
        self._samples.append((sample, propagation_distance.simplified))

    def conduct(self):
        """Conduct the experiment."""
        pass

    def create_geometry(self):
        pass

    def propagate(self):
        pass

    def detect(self):
        pass
