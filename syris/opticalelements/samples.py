"""Sample constitutes of graphical objects and to them assigned materials."""


class Sample(object):

    """Base class for a sample."""

    def __init__(self, material):
        self.material = material


class MovingSample(Sample):

    """A sample consisting of moving graphical objects."""

    def __init__(self, material, gr_object):
        """Create a moving sample composed of graphical object *gr_object*.
        It can be a :py:class:`CompositeObject` or :py:class:`MetaObject`.
        """
        super(MovingSample, self).__init__(material)
        self.object = gr_object


class StaticSample(Sample):

    """A sample which does not move throughout an experiment."""

    def __init__(self, material, thickness):
        """Create a static sample with projected *thickness*."""
        super(StaticSample, self).__init__(material)
        self.thickness = thickness
