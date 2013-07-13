"""Detector is composed of a camera and a lens."""


class Detector(object):

    """A detector consisting of a camera and an objective lens."""

    def __init__(self, lens, camera):
        """Create a detector with *lens* and a *camera*."""
        self.lens = lens
        self.camera = camera

    @property
    def pixel_size(self):
        return self.camera.pixel_size.simplified / self.lens.magnification
