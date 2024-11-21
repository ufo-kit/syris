import numpy as np
import pyvista as pv
from Quaternion import RotationQuaternion as rq
from CoordinateSystem import CoordinateSystem as C
import quantities as pq




def is_length(param):
    return isinstance(param, pq.Quantity) and param.dimensionality.simplified == pq.m.dimensionality.simplified

class CameraCoordinates:
    pass