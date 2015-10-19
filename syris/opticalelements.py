"""
Module contains optical elements which consist of a graphical object and a material.
"""
from syris.physics import transfer, energy_to_wavelength


class OpticalElement(object):

    """An optical element consisting of a :class:`~syris.graphicalobjects.GraphicalObject` and a
    :class:`~syris.materials.Material`.
    """

    def __init__(self, graphical_object, material):
        self.graphical_object = graphical_object
        self.material = material

    def transfer(self, energy, t=None, queue=None, out=None):
        """Compute the transfer function at *energy*, time *t*. Use *queue* for OpenCL computations
        and *out* pyopencl array.
        """
        ri = self.material.get_refractive_index(energy)
        lam = energy_to_wavelength(energy)

        return transfer(self.graphical_object.project(t=t), ri, lam, queue=queue, out=out)
