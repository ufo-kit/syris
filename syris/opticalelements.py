"""
Module contains optical elements which consist of a graphical object and a material.
"""
import quantities as q
from syris.physics import transfer, energy_to_wavelength


class OpticalElement(object):

    """An optical element consisting of a :class:`~syris.graphicalobjects.GraphicalObject` and a
    :class:`~syris.materials.Material`.
    """

    def __init__(self, graphical_object, material):
        self.graphical_object = graphical_object
        self.material = material

    def transfer(self, shape, pixel_size, energy, t=0 * q.s, queue=None, out=None):
        """Compute the transfer function of the projected thickness on an image plane of size
        *shape* (y, x), use *pixel_size*, *energy*, time *t*. Use *queue* for OpenCL computations
        and *out* pyopencl array.
        """
        ri = self.material.get_refractive_index(energy)
        lam = energy_to_wavelength(energy)

        return transfer(self.graphical_object.project(shape, pixel_size, t=t),
                        ri, lam, queue=queue, out=out)
