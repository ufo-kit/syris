import cupy as cp
from cupy.cuda import Event, Stream, get_elapsed_time

cp.clear_memo()


class CudaTimer:
    def __init__(self, stream: Stream):
        self._start = Event()
        self._stop = Event()
        self._stream = stream
        self._isRunning = False

    def start(self):
        self._start.record(self._stream)
        self._isRunning = True

    def stop(self):
        self._stop.record(self._stream)
        self._isRunning = False

    def elapsedTime(self) -> float:
        assert not self._isRunning, "CudaTimer is still running"
        self._stop.synchronize()
        time = get_elapsed_time(self._start, self._stop)  # ms
        return time

    def reset(self):
        self._start = Event()
        self._stop = Event()
        self._isRunning = False


class CudaPipeline:
    """
    Manage and run CUDA kernels using CuPy.
    """

    def __init__(self, headers: list, options: list = None):
        self.modules = {}
        self.kernels = {}
        self.opts = ["-I " + h + " " for h in headers]
        self._stream = cp.cuda.Stream()
        self.timer = CudaTimer(self._stream)
        if options is not None:
            self.opts += options

    def readModuleFromFiles(
        self,
        moduleName: str,
        fileNames: list,
        options: list = None,
        name_expressions: list = None,
        backend: str = "nvcc",
        jitify: bool = False,
    ):
        if moduleName in self.modules:
            raise Exception("Module already loaded")

        if options is None:
            selected_options = self.opts
        else:
            selected_options = options + self.opts

        selected_options += ["-D__CUDA_NO_HALF_CONVERSIONS__", "--std=c++17"]

        selected_options = tuple(
            selected_options,
        )

        # Prepend
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
            backend=backend,
        )

    def getKernelFromModule(self, moduleName: str, kernelName: str) -> cp.RawKernel:
        if moduleName not in self.modules:
            raise Exception("Module not found")

        if kernelName not in self.kernels:
            self.kernels[kernelName] = self.modules[moduleName].get_function(kernelName)

        return self.kernels[kernelName]

    def synchronize(self):
        self._stream.synchronize()

    def timeit(func):
        def wrapper(self, *args, **kwargs):
            self.timer.start()
            result = func(self, *args, **kwargs)
            self.synchronize()
            self.timer.stop()
            t = self.timer.elapsedTime()
            self.timer.reset()
            return (t, result)

        return wrapper

    @timeit
    def launchKernel(self, kernelName: str, *args):
        if kernelName not in self.kernels:
            raise Exception("Kernel not found")
        return self.kernels[kernelName](*args)
