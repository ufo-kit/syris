import numpy as np
import logging

LOG = logging.getLogger(__name__)

# --- Library Availability Detection ---
try:
    import cupy as cp
    # A simple check to ensure a device is actually available
    if cp.cuda.runtime.getDeviceCount() > 0:
        HAS_CUPY = True
    else:
        HAS_CUPY = False
        cp = None
except ImportError:
    HAS_CUPY = False
    cp = None

try:
    import pyopencl as cl
    if cl.get_platforms():
        HAS_OPENCL = True
    else:
        HAS_OPENCL = False
        cl = None
except ImportError:
    HAS_OPENCL = False
    cl = None

from .bodies.accelerators import BvhCupyAccelerator, LegacyCpuAccelerator

class ComputeBackend:
    """Detects and configures the compute engine (CUDA or OpenCL)."""
    CUDA = 'cuda'
    OPENCL = 'opencl'
    NONE = 'none'

    def __init__(self, compute_backend):
        if HAS_CUPY and compute_backend == self.CUDA:
            self.name = self.CUDA
            self.xp = cp
            self.pipeline = None
            LOG.info("CUDA compute engine selected.")
        else:
            if HAS_OPENCL:
                self.name = self.OPENCL
                self.xp = np
                self.queue = None
                LOG.info("OpenCL compute engine selected.")
            else:
                LOG.warning("No compute backend available.")

    def get_accelerator_for_mesh(self, mesh):
        """Factory method to return the correct accelerator instance."""
        if self.name == self.CUDA:
            return BvhCupyAccelerator(mesh)
        elif self.name == self.OPENCL:
            return LegacyCpuAccelerator(mesh)
        else:
            return LegacyCpuAccelerator(mesh)
