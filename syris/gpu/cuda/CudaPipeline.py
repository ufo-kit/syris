import cupy as cp
import numpy as np

float4 = cp.dtype({
            'names': ['x', 'y', 'z', 'w'],
            'formats': [cp.float32] * 4,
        })

def make_float4(x, y, z, w) -> cp.ndarray:
    try:
        return cp.array([x, y, z, w]).astype(cp.float32).view(float4)
    except ValueError:
        pass

class CudaPipeline:
    def __init__(self, headers : list, options : list = None):
        self.modules = {}
        self.kernels = {}
        self.opts = ["-I " + h + " " for h in headers]
        if options is not None:
            self.opts += options
    
    def readModuleFromFiles(self, 
        moduleName : str,
        fileNames : list, 
        options : list = None,
        name_expressions : list = None,
        backend : str = "nvcc",
        jitify : bool = False):
        if moduleName in self.modules:
            raise Exception("Module already loaded")
    
        if options is None:
            selected_options = self.opts
        else:
            selected_options = options + self.opts
        
        selected_options += ['-D__CUDA_NO_HALF_CONVERSIONS__']
        
        selected_options = tuple(selected_options,)
        print (selected_options)

        # Prepend cub header
        code = r"""
        #include <cub/cub.cuh>
        #include <thrust/sort.h>
        #include <thrust/device_vector.h>
        #include <thrust/execution_policy.h>
        """
        for fileName in fileNames:
            with open(fileName, "r") as f:
                source = f.read()
                code += source + "\n"
        self.modules[moduleName] = cp.RawModule(
            code=code,
            options=selected_options,
            jitify=jitify,
            name_expressions=name_expressions,
            backend=backend)

    def getKernelFromModule(self, moduleName : str, kernelName : str) -> cp.RawKernel:
        if moduleName not in self.modules:
            raise Exception("Module not found")
        
        if kernelName not in self.kernels:
            self.kernels[kernelName] = self.modules[moduleName].get_function(kernelName)
        
        return self.kernels[kernelName]

    