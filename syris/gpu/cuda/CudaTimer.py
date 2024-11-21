from cupy.cuda import Event, Stream, get_elapsed_time

class CudaTimer:
    def __init__(self, stream : Stream):
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
        time = get_elapsed_time(self._start, self._stop) # ms
        return time

    def reset(self):
        self._start = Event()
        self._stop = Event()
        self._isRunning = False
