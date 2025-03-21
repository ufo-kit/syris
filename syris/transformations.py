import numpy as np
import quantities as q

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

class Backend:
    NUMPY = 'numpy'
    CUPY = 'cupy'
    
    # Map user-friendly names to internal backend names
    _aliases = {
        "cpu": NUMPY,
        "gpu": CUPY,
        "numpy": NUMPY,
        "cupy": CUPY,
    }
    
    def __init__(self, device=None):
        if device is None:
            device = self.CUPY

        # Convert the provided string to lower case and map it using aliases.
        device = device.lower()
        device = self._aliases.get(device, device)
        if device == self.CUPY and not HAS_CUPY:
            print("Warning: CuPy not available, falling back to NumPy")
            self._current = self.NUMPY
        else:
            self._current = device

        self.clear_memory()

    def get(self):
        if self._current == self.CUPY and not HAS_CUPY:
            print("Warning: CuPy not available, falling back to NumPy")
            return np
        return cp if self._current == self.CUPY else np

    def set(self, device):
        device = device.lower()
        device = self._aliases.get(device, device)
        if device == self.CUPY and not HAS_CUPY:
            print("Warning: CuPy not available, using NumPy instead")
            self._current = self.NUMPY
        else:
            self._current = device

    def use_gpu(self):
        self.set("gpu")
    
    def use_cpu(self):
        self.set("cpu")

    def clear_memory(self):
        if self._current == self.CUPY:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def to_numpy(self, array):
        xp = self.get()
        return array.get() if xp is cp else array

    def to_device(self, array):
        xp = self.get()
        return xp.asarray(array)

class Quaternion:
    """
    A class representing a quaternion.
    """
    def __init__(self, w, x, y, z):
        self.q = np.array([w, x, y, z], dtype=np.float32)
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
        cross = np.cross(np.asarray(vec1.imag()), np.asarray(vec2.imag()))
        return Quaternion(0, *cross)

    @staticmethod
    def dot_product(vec1, vec2):
        dot = np.dot(np.asarray(vec1.imag()), np.asarray(vec2.imag()))
        return Quaternion(dot, 0, 0, 0)
    
    @staticmethod
    def scalar_product(vec, scalar):
        scaled = np.asarray(vec.q) * scalar
        return Quaternion(*scaled)
    
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
        # Extracting components in a vectorized way
        w1, v1 = self.q[0], self.q[1:]
        w2, v2 = other.q[0], other.q[1:]
        
        # Computing the components of the resulting quaternion
        w = w1 * w2 - np.dot(v1, v2)
        v = w1 * v2 + w2 * v1 + np.cross(v1, v2)
        
        # Create resulting quaternion
        result = np.zeros(4, dtype=np.float32)
        result[0] = w
        result[1:] = v
        
        return Quaternion(*result)
    
    def __truediv__(self, scalar):
        return Quaternion(*(self.q / scalar))
    
    def __str__(self):
        return f"({self.q[0]}, {self.q[1]}, {self.q[2]}, {self.q[3]})"

class RotationQuaternion(Quaternion):
    """
    A class representing a rotation quaternion.
    Non vectorized version.
    """
    def __init__(self, angle, axis):
        if hasattr(angle, 'rescale'):
            angle_rad = angle.rescale(q.rad).magnitude
        else:
            angle_rad = angle
        self.angle = angle_rad
        half_angle = angle_rad / 2
        if len(axis) == 4:
            axis = axis[:3]
        sin_half_angle = np.sin(half_angle)
        axis_norm = np.linalg.norm(axis)
        w, x, y, z = [np.cos(half_angle), 
                    axis[0]*sin_half_angle/axis_norm,
                    axis[1]*sin_half_angle/axis_norm,
                    axis[2]*sin_half_angle/axis_norm]
        super().__init__(w, x, y, z)
        self.normalized = True
    
    @property
    def matrix(self):
        """
        Homogeneous Orthogonal Matrix representation of the quaternion.
        """
        if not self.normalized:
            self.normalize()
        w, x, y, z = self.q
        return np.array([
            [w*w + x*x - y*y - z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y, 0],
            [2*x*y + 2*w*z, w*w - x*x + y*y - z*z, 2*y*z - 2*w*x, 0],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, w*w - x*x - y*y + z*z, 0],
            [0, 0, 0, 1]
        ])
    
    def rotate(self, v):
        # Combine scalar zero with the vector into a single GPU array
        q_input = np.concatenate([np.zeros(1, dtype=v.dtype), v], axis=0)
        q = Quaternion(*q_input)
        lhs = self
        rhs = self.inverse() if not self.normalized else self.conjugate()
        rotated_vector = (lhs * q * rhs).imag()
        
        try:
            return rotated_vector * v.units
        except AttributeError:
            return rotated_vector
        

class Transformation(object):
    """
    A class representing a transformation matrix. Uses backend to determine whether to use NumPy or CuPy.
    """
    def __init__(self, matrix=None, device=None, dtype=None):
        self.backend = Backend(device)
        self._xp = self.backend.get()

        if dtype is None:
            self.dtype = np.float32
        else:
            self.dtype = dtype

        if matrix is None:
            self.matrix = self._xp.eye(4, dtype=self.dtype)
        else:
            self.matrix = self._xp.asarray(matrix, dtype=self.dtype)

    def reset(self):
        """
        Reset the transformation matrix.
        """
        self.matrix = self._xp.eye(4, dtype=self.dtype)
        self.backend.clear_memory()

    def apply(self, points):
        """
        Apply transformation to an array of points.
        Points can be a single point [x, y, z] or an array of points [[x1, y1, z1], ...].
        """
        # Store original units if present.
        has_units = hasattr(points, 'units')
        units = getattr(points, 'units', None)
        
        xp = self._xp
        points = self.backend.to_device(points)

        # Ensure self.matrix is on the same device
        matrix = self.backend.to_device(self.matrix)

        if points.ndim == 1:
            # Pre-allocate homogeneous coordinate array.
            homogeneous = xp.ones(4, dtype=self.dtype)
            homogeneous[:3] = points
            result = matrix @ homogeneous

            w = result[3]
            if not xp.isclose(w, 0) and w != 1:
                result[:3] /= w
            transformed = result[:3]
        else:
            num_points = points.shape[0]
            homogeneous = xp.ones((num_points, 4), dtype=self.dtype)
            homogeneous[:, :3] = points

            result = homogeneous @ matrix.T

            # Avoid division by near-zero values
            w = result[:, 3]  # Make w a 1D array
            mask = xp.abs(w) > 1e-10
            transformed = xp.empty_like(result[:, :3])
            transformed[mask] = result[mask, :3] / w[mask, None]  # Add None for broadcasting
            transformed[~mask] = result[~mask, :3]

        result_np = self.backend.to_numpy(transformed)
        return result_np * units if has_units else result_np
    
    def compose(self, other):
        """
        Compose this transformation with another transformation.
        Returns a new transformation that applies this transformation followed by the other.
        """
        xp = self._xp
        result_matrix = xp.dot(other.matrix, self.matrix)
        return Transformation(result_matrix, device=self.backend._current)
        
    def inverse(self):
        """
        Return the inverse transformation.
        """
        xp = self._xp
        inv_matrix = xp.linalg.inv(self.matrix)
        return Transformation(inv_matrix, device=self.backend._current)
        
    def translation(self, x=0.0, y=0.0, z=0.0):
        """
        Apply a translation transformation to the current matrix.
        """
        xp = self._xp
        trans_matrix = xp.eye(4, dtype=self.dtype)
        trans_matrix[0, 3] = x
        trans_matrix[1, 3] = y
        trans_matrix[2, 3] = z
        self.matrix = xp.dot(trans_matrix, self.matrix)
        return self
        
    def from_vector(self, v):
        """
        Apply a translation transformation from a vector.
        """
        v = self.backend.to_device(v)
        return self.translation(v[0], v[1], v[2])
        
    def rotation_x(self, angle):
        """
        Apply a rotation around the x-axis.
        """
        angle_rad = angle.rescale(q.rad).magnitude if hasattr(angle, 'rescale') else angle
        c, s = self._xp.cos(angle_rad), self._xp.sin(angle_rad)
        rot_matrix = self._xp.eye(4, dtype=self.dtype)
        rot_matrix[1, 1] = c
        rot_matrix[1, 2] = -s
        rot_matrix[2, 1] = s
        rot_matrix[2, 2] = c
        self.matrix = self._xp.dot(rot_matrix, self.matrix)
        return self
        
    def rotation_y(self, angle):
        """
        Apply a rotation around the y-axis.
        """
        angle_rad = angle.rescale(q.rad).magnitude if hasattr(angle, 'rescale') else angle
        c, s = self._xp.cos(angle_rad), self._xp.sin(angle_rad)
        rot_matrix = self._xp.eye(4, dtype=self.dtype)
        rot_matrix[0, 0] = c
        rot_matrix[0, 2] = s
        rot_matrix[2, 0] = -s
        rot_matrix[2, 2] = c
        self.matrix = self._xp.dot(rot_matrix, self.matrix)
        return self
        
    def rotation_z(self, angle):
        """
        Apply a rotation around the z-axis.
        """
        angle_rad = angle.rescale(q.rad).magnitude if hasattr(angle, 'rescale') else angle
        c, s = self._xp.cos(angle_rad), self._xp.sin(angle_rad)
        rot_matrix = self._xp.eye(4, dtype=self.dtype)
        rot_matrix[0, 0] = c
        rot_matrix[0, 1] = -s
        rot_matrix[1, 0] = s
        rot_matrix[1, 1] = c
        self.matrix = self._xp.dot(rot_matrix, self.matrix)
        return self
        
    def rotation(self, angle, axis):
        """
        Apply a rotation around an arbitrary axis.
        """
        if hasattr(angle, 'rescale'):
            angle_rad = angle.rescale(q.rad).magnitude
        else:
            angle_rad = angle
        rot_matrix = RotationQuaternion(angle_rad, axis).matrix
        self.matrix = self._xp.dot(self._xp.asarray(rot_matrix), self.matrix)
        return self
        
    def scaling(self, sx=1.0, sy=1.0, sz=1.0):
        """
        Apply a scaling transformation.
        """
        scale_matrix = self._xp.eye(4, dtype=self.dtype)
        scale_matrix[0, 0] = sx
        scale_matrix[1, 1] = sy
        scale_matrix[2, 2] = sz
        self.matrix = self._xp.dot(scale_matrix, self.matrix)
        return self
        
    def uniform_scaling(self, scale):
        """
        Apply a uniform scaling transformation.
        """
        return self.scaling(scale, scale, scale)
        
    def shear_xy(self, angle):
        """
        Apply a shear transformation in the xy plane.
        """
        angle_rad = angle.rescale(q.rad).magnitude if hasattr(angle, 'rescale') else angle
        shear_matrix = self._xp.eye(4, dtype=self.dtype)
        shear_matrix[0, 1] = self._xp.tan(angle_rad)
        self.matrix = self._xp.dot(shear_matrix, self.matrix)
        return self
        
    def shear_xz(self, angle):
        """
        Apply a shear transformation in the xz plane.
        """
        angle_rad = angle.rescale(q.rad).magnitude if hasattr(angle, 'rescale') else angle
        shear_matrix = self._xp.eye(4, dtype=self.dtype)
        shear_matrix[0, 2] = self._xp.tan(angle_rad)
        self.matrix = self._xp.dot(shear_matrix, self.matrix)
        return self
        
    def shear_yx(self, angle):
        """
        Apply a shear transformation in the yx plane.
        """
        angle_rad = angle.rescale(q.rad).magnitude if hasattr(angle, 'rescale') else angle
        shear_matrix = self._xp.eye(4, dtype=self.dtype)
        shear_matrix[1, 0] = self._xp.tan(angle_rad)
        self.matrix = self._xp.dot(shear_matrix, self.matrix)
        return self
        
    def shear_yz(self, angle):
        """
        Apply a shear transformation in the yz plane.
        """
        angle_rad = angle.rescale(q.rad).magnitude if hasattr(angle, 'rescale') else angle
        shear_matrix = self._xp.eye(4, dtype=self.dtype)
        shear_matrix[1, 2] = self._xp.tan(angle_rad)
        self.matrix = self._xp.dot(shear_matrix, self.matrix)
        return self
        
    def shear_zx(self, angle):
        """
        Apply a shear transformation in the zx plane.
        """
        angle_rad = angle.rescale(q.rad).magnitude if hasattr(angle, 'rescale') else angle
        shear_matrix = self._xp.eye(4, dtype=self.dtype)
        shear_matrix[2, 0] = self._xp.tan(angle_rad)
        self.matrix = self._xp.dot(shear_matrix, self.matrix)
        return self
        
    def shear_zy(self, angle):
        """
        Apply a shear transformation in the zy plane.
        """
        angle_rad = angle.rescale(q.rad).magnitude if hasattr(angle, 'rescale') else angle
        shear_matrix = self._xp.eye(4, dtype=self.dtype)
        shear_matrix[2, 1] = self._xp.tan(angle_rad)
        self.matrix = self._xp.dot(shear_matrix, self.matrix)
        return self
        
    def look_at(self, eye, target, up=[0, 1, 0]):
        """
        Apply a viewing transformation that looks from eye to target.
        """
        eye = self.backend.to_device(eye)
        target = self.backend.to_device(target)
        up = self.backend.to_device(up)
        
        # Calculate the forward vector (z)
        forward = eye - target
        forward = forward / self._xp.linalg.norm(forward)
        
        # Calculate the right vector (x)
        right = self._xp.cross(up, forward)
        right = right / self._xp.linalg.norm(right)
        
        # Recalculate the up vector (y)
        up = self._xp.cross(forward, right)
        
        # Create the rotation part
        view_matrix = self._xp.eye(4, dtype=self.dtype)
        view_matrix[0, :3] = right
        view_matrix[1, :3] = up
        view_matrix[2, :3] = forward
        
        # Create the translation part
        translation = self._xp.zeros(3)
        translation[0] = -self._xp.dot(right, eye)
        translation[1] = -self._xp.dot(up, eye)
        translation[2] = -self._xp.dot(forward, eye)
        
        view_matrix[0:3, 3] = translation
        
        self.matrix = self._xp.dot(view_matrix, self.matrix)
        return self
        
    def __str__(self):
        matrix_np = self.backend.to_numpy(self.matrix)
        return f"Transformation(\n{matrix_np}\n)"


