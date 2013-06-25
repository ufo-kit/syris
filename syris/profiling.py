import pyopencl as cl
from threading import Thread
from Queue import Queue, Empty
import itertools
import logging


logger = logging.getLogger(__name__)

# Singleton.
profiler = None

class Profiler(Thread):
    states = {  cl.profiling_info.QUEUED : "QUEUED",
                cl.profiling_info.SUBMIT : "SUBMIT",
                cl.profiling_info.START : "START",
                cl.profiling_info.END : "END"
              }
    format_string = "%d\t%d\t%d\t%s\t%s\t%d"
    
    def __init__(self, file_name):
        Thread.__init__(self)
        self._profile_file = open(file_name, "w")
        self._events = Queue()
        self._event_next = itertools.count().next
        self._clqeue_next = itertools.count().next
        self._cldevice_next = itertools.count().next
        self._clqueues = {}             # {queue: id}
        self._cldevices = {}            # {device: id}
        self.daemon = True
        self.finish = False
        
        self._profile_file.write("# "+
             Profiler.format_string.replace("%d", "%s")
             % ("event_id", "command_queue_id", "device_id",
                "state", "func_name", "time") +"\n")            
        
    def run(self):
        while not self.finish or not self._events.empty():
            try:
                event, kernel = self._events.get(timeout=0.1)
                cl.wait_for_events([event])
                self._process(event, kernel)
                self._events.task_done()                
            except Empty :
                pass
            except Exception as e:
                logger.error(e.message)
            finally:
                self._profile_file.close()
        
    def shutdown(self):
        """Wait for all events to finish and then stop the profiler loop."""
        self.finish = True
        self.join()
        logger.info("Profiler finished.")
            
    def add(self, event, func_name=""):
        self._events.put((event, func_name))
        
    def _get_string(self, state, event_id, clq_id, device_id, func_name, t):
        """Format profile string.
        
        @param state: Event state
        @param event_id: event's id
        @param clq_id: command queue
        @param func_name: kernel function name
        @param t: time [ns] 
        """
        return Profiler.format_string % (event_id, clq_id, device_id,
                                   Profiler.states[state], func_name, t)
        
    def _process(self, event, func_name=""):
        # clqueue id
        if event.command_queue not in self._clqueues:
            self._clqueues[event.command_queue] = self._clqeue_next()
            
        if event.command_queue.device not in self._cldevices:
            self._cldevices[event.command_queue.device] = self._cldevice_next()
        
        if func_name == "":
            func_name = "N/A"
        
        event_id = self._event_next()
        s = "%s\n%s\n%s\n%s\n" % (self._get_string(cl.profiling_info.QUEUED,
                   event_id, self._clqueues[event.command_queue],
                   self._cldevices[event.command_queue.device], func_name,
                                     event.profile.QUEUED),
                                  
                self._get_string(cl.profiling_info.SUBMIT, event_id,
                 self._clqueues[event.command_queue],
                 self._cldevices[event.command_queue.device], func_name,
                 event.profile.SUBMIT),
                                  
                self._get_string(cl.profiling_info.START, event_id,
                 self._clqueues[event.command_queue],
                 self._cldevices[event.command_queue.device], func_name,
                 event.profile.START),
                                  
                self._get_string(cl.profiling_info.END, event_id,
                 self._clqueues[event.command_queue],
                 self._cldevices[event.command_queue.device], func_name,
                 event.profile.END))
        
        self._profile_file.write(s)