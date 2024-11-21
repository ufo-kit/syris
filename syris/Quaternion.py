import numpy as np
import quantities as pq

class Quaternion:
    def __init__(self, w, x, y, z):
        self.q = np.array([w, x, y, z])
        self.normalized = False
        self._inverse = None

    def real(self):
        return self.q[0]

    def imag(self):
        return self.q[1:] 
    
    def conjugate(self):
        return Quaternion(self.q[0], -self.q[1], -self.q[2], -self.q[3])
    
    @staticmethod
    def cross_product(vec1, vec2):
        cross = np.cross(vec1.imag(), vec2.imag())
        return Quaternion(0, *cross)

    @staticmethod
    def dot_product(vec1, vec2):
        dot = np.dot(vec1.imag(), vec2.imag())
        return Quaternion(dot, 0, 0, 0)
    
    @staticmethod
    def scalar_product(vec, scalar):
        return Quaternion(*(vec.q * scalar))
    
    def norm(self):
        return np.linalg.norm(self.q)

    def normalize(self):
        norm = self.norm()
        self.q /= norm
        self.normalized = True
        return self

    def inverse(self):
        if self._inverse is None:
            self._inverse = self.conjugate() / self.norm()
        return self._inverse

    def __add__(self, other):
        return Quaternion(*(self.q + other.q))
    
    def __sub__(self, other):
        return Quaternion(*(self.q - other.q))
    
    def __mul__(self, other):
        w1, x1, y1, z1 = self.q
        w2, x2, y2, z2 = other.q
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return Quaternion(w, x, y, z)
    
    def __truediv__(self, scalar):
        return Quaternion(*(self.q / scalar))
    
    def __str__(self):
        return f"({self.q[0]}, {self.q[1]}, {self.q[2]}, {self.q[3]})"
    

class RotationQuaternion(Quaternion):
    def __init__(self, angle, axis):
        self.angle = angle.rescale(pq.rad)

        half_angle = self.angle / 2
        
        if len(axis) == 4:
            axis = axis[:3]

        sin_half_angle = np.sin(half_angle)
        axis_norm = np.linalg.norm(axis)

        super().__init__(np.cos(half_angle), *(axis * sin_half_angle / axis_norm))

        self.normalized = True
    
    def rotate(self, v):
        q = Quaternion(0, *v)
        lhs = self
        if not self.normalized:
            rhs = self.inverse()
        else:
            rhs = self.conjugate()

        rotated_vector = (lhs * q * rhs).imag()
        
        try:
            return rotated_vector * v.units
        except AttributeError:
            return rotated_vector
