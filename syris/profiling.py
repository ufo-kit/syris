"""
Module for profiling OpenCL GPU code execution.
"""

import pyopencl as cl
from threading import Thread
from Queue import Queue, Empty
import quantities as q
import itertools
import logging
import time


logger = logging.getLogger(__name__)


class DummyProfiler():
    """A profiler which does nothing for saving time."""
    def add(self, event, func_name=""):
        pass


class Profiler(Thread):
    """An OpenCL GPU code profiler."""
    states = [cl.profiling_info.QUEUED, cl.profiling_info.SUBMIT,
              cl.profiling_info.START, cl.profiling_info.END]
    state_strs = dict(zip(states, ["QUEUED", "SUBMIT", "START", "END"]))
    format_string = "%d\t%d\t%d\t%s\t%s\t%d"

    def __init__(self, queues, file_name):
        """Create a profiler for command *queues* and output file
        *file_name*.
        """
        Thread.__init__(self)
        self._events = Queue()
        self._event_next = itertools.count().next
        self._clqeue_next = itertools.count().next
        self._cldevice_next = itertools.count().next
        self._clqueues = {}             # {queue: id}
        self._cldevices = {}            # {device: id}
        self.daemon = True
        self.finish = False
        self._file_name = None
        self._profile_file = None
        self._file_name = file_name

        self._profile_file = open(file_name, "w")
        self._profile_file.write("# units\nns\n")
        self._write_time_shift(queues)

        self._profile_file.write("# " +
                                 Profiler.format_string.replace("%d", "%s")
                                 % ("event_id", "command_queue_id",
                                    "device_id", "state", "func_name",
                                    "time") + "\n")

    def _write_time_shift(self, queues):
        # Get only a single command queue for a device on which we will
        # determine the zero time of a device.
        unique_queues = []
        devices = []
        for queue in queues:
            if queue.device not in devices:
                unique_queues.append(queue)
                devices.append(queue.device)

        starts = {}
        st = time.time()
        for i in range(len(devices)):
            starts[devices[i]] = cl.enqueue_marker(unique_queues[i])
        dt = (time.time()-st)*q.s

        cl.wait_for_events(starts.values())

        for device in starts:
            starts[device] = starts[device].profile.queued

        # Write the zero time for every device into the profiling file.
        self._profile_file.write("# device\tinitial_time\n")
        for device in starts:
            self._cldevices[device] = self._cldevice_next()
            self._profile_file.write("%d\t%d\n" % (self._cldevices[device],
                                                   starts[device]))
        self._profile_file.write("# END_INIT_T0\n")
        self._profile_file.write("# Relative device timing error\n%g\n" %
                                 dt.rescale(q.ns))
        self._profile_file.write("# END_INIT\n")

    def run(self):
        while not self.finish or not self._events.empty():
            try:
                event, kernel = self._events.get(timeout=0.1)
                cl.wait_for_events([event])
                self._process(event, kernel)
                self._events.task_done()
            except Empty:
                pass
            except Exception as e:
                logger.error(e)
        self._profile_file.close()

    def shutdown(self):
        """Wait for all events to finish and then stop the profiler loop."""
        self.finish = True
        self.join()
        logger.info("Profiler finished.")

    def add(self, event, func_name=""):
        """Add an OpenCL *event* and function with name *func_name*
        into the profiler's queue.
        """
        self._events.put((event, func_name))

    def _get_string(self, event_state, event, event_id, func_name):
        """Format profile string based on *event_state*, *event_id*
        and *func_name*.
        """
        return Profiler.format_string % (event_id,
                                         self._clqueues[event.command_queue],
                                         self._cldevices[
                                         event.command_queue.device],
                                         Profiler.state_strs[event_state],
                                         func_name,
                                         event.get_profiling_info(event_state))

    def _process(self, event, func_name=""):
        # clqueue id
        if event.command_queue not in self._clqueues:
            self._clqueues[event.command_queue] = self._clqeue_next()

        if event.command_queue.device not in self._cldevices:
            raise RuntimeError("%s not in devices list." %
                               (str(event.command_queue.device)))

        if func_name == "":
            func_name = "N/A"

        event_id = self._event_next()

        # Get all records for one event (queued, submit, ...)
        strings = [self._get_string(event_state, event, event_id, func_name)
                   for event_state in Profiler.states]
        s = "%s\n%s\n%s\n%s\n" % (strings[0], strings[1], strings[2],
                                  strings[3])

        self._profile_file.write(s)

# Singleton.
profiler = None


#=============================================================================
# Profiling Reconstruction
#=============================================================================


import re
import sys
from matplotlib import pyplot as plt
from optparse import OptionParser
import os

colors = {
    "b": "blue",
    "g": "green",
    "r": "red",
    "c": "cyan",
    "m": "magenta",
    "y": "yellow",
    "k": "black",
    "w": "white"
}


class _Record(object):
    def __init__(self, *args, **kwargs):
        for a, v in args:
            setattr(self, a, v)

    def __str__(self):
        return "Record("+str(self.__dict__)+")"


class _Event(object):
    def __str__(self):
        return "Event("+str(self.__dict__)+")"


class ProfileReconstructor(object):
    """Profile reconstructor which handles the profiling file created by
    :py:class:`Profiler`.
    """
    attributes = ["EVENT_ID", "QUEUE_ID", "DEVICE_ID", "STATE", "FUNC_NAME",
                  "TIME"]
    pattern = re.compile(r"(?P<%s>[0-9]+)\s*(?P<%s>[0-9]+)\s*(?P<%s>[0-9]+)" %
                         (attributes[0], attributes[1], attributes[2]) +
                         "\s*(?P<%s>[A-Z]+)\s*(?P<%s>\w*)\s*(?P<%s>[0-9]+)\n" %
                         (attributes[3], attributes[4], attributes[5]))
    cl_states = ["QUEUED", "SUBMIT", "START", "END"]
    str_to_qtime = {q.ns.symbol: q.ns,
                    q.us.symbol: q.us,
                    q.ms.symbol: q.ms,
                    q.s.symbol: q.s}

    def __init__(self, file_name, str_units):
        """Create profile reconstructor reading from *file_name* and using
        units based on string *str_units*.
        """
        self._profile_file_name = file_name
        self._min_time = sys.maxint
        self._records = []
        self._events = {}
            # event objects dictionary {event_id: event}
        self._starts = {}           # {device_id: initial_time}
        self._cache = {}, None
        self.units = ProfileReconstructor.str_to_qtime[str_units]
        # We will need to convert between units used in the file and
        # units wanted for output.
        self.file_units = None

    def get_data(self, attr):
        """Get data in a dictionary aggregated to a multidictionary. *attr*
        is a record attribute which will serve as a key to the top level
        of result dictionary Return a dictionary in form
        {attr: {event_id: Event}}.
        """
        if attr == self._cache[1]:
            # last request was on the same attribute -> data do not change
            return self._cache[0]

        if self._events == {}:
            self._process()
        self._cache = self._get_aggregate(attr), attr

        return self._cache[0]

    def _analyze_line(self, s):
        m = ProfileReconstructor.pattern.search(s)

        vals = [int(m.group(ProfileReconstructor.attributes[0])),
                int(m.group(ProfileReconstructor.attributes[1])),
                int(m.group(ProfileReconstructor.attributes[2])),
                m.group(ProfileReconstructor.attributes[3]),
                m.group(ProfileReconstructor.attributes[4]),
                int(m.group(ProfileReconstructor.attributes[5]))
                ]
        args = zip(ProfileReconstructor.attributes, vals)

        return _Record(*args)

    def _process_header(self, f):
        self._process_units(f)
        self._process_t0(f)
        self._process_dt_error(f)

    def _process_units(self, f):
        l = f.readline()
        if l != "# units\n":
            raise ValueError("File corrupted.")
        l = f.readline().strip()
        self.file_units = ProfileReconstructor.str_to_qtime[l.lower()]

    def _process_t0(self, f):
        """Different devices have different time offsets,
        make the calibration.
        """
        l = ""
        while l != "# END_INIT_T0\n":
            l = f.readline()
            if not l.startswith("#"):
                device_id, device_start = l.strip().split("\t")
                self._starts[int(device_id)] = int(device_start)

    def _process_dt_error(self, f):
        """Error estimate for different device zero times with respect
        to global zero.
        """
        l = ""
        while l != "# END_INIT\n":
            l = f.readline()
            if not l.startswith("#"):
                dt = float(l.strip())
                print "Relative device time error: %g %s" %\
                    (q.Quantity(dt, self.file_units).rescale(self.units),
                     self.units.symbol)

    def _process(self):
        records = {}
        with open(self._profile_file_name, "r") as f:
            self._process_header(f)
            # skip the first one (explanation line)
            line = f.readline()
            while True:
                line = f.readline()
                if line == "":
                    break
                rec = self._analyze_line(line)
                self._records.append(rec)
                if rec.EVENT_ID not in records:
                    records[rec.EVENT_ID] = []
                records[rec.EVENT_ID].append(rec)

        for rec in self._records:
            rec.TIME -= self._starts[rec.DEVICE_ID]

        self._create_events(records)

    def _create_events(self, records):
        """Create events from individual *records* for each event based on
        computational status (QUEUED, SUBMIT, START, END).
        """
        for event_records in records.values():
            for rec in event_records:
                if rec.EVENT_ID not in self._events:
                    self._events[rec.EVENT_ID] = _Event()
                for k in rec.__dict__:
                    if k == "STATE":
                        setattr(self._events[rec.EVENT_ID], getattr(rec, k),
                                getattr(rec, "TIME"))
                    elif k != "TIME":
                        setattr(self._events[rec.EVENT_ID], k, getattr(rec, k))

    def _get_aggregate(self, attribute):
        prop = {}
        for event_id in self._events:
            if getattr(self._events[event_id], attribute) not in prop:
                prop[getattr(self._events[event_id], attribute)] = []
            prop[getattr(self._events[event_id], attribute)
                 ].append(self._events[event_id])

        return prop


def plot(data, attribute, states, file_units, out_units, start_from=0,
         stop_at=sys.float_info.max, delta=0.0, only_averages=False):
    """Plot the profiling information, where

    * *data* - a dictionary in the form {id: {event_id: values}}
    * *attribute* - (event_id, device_id, queue_id)
    * *states* - OpenCL Event states to use as beginning and end
    * *file_units* - units used in the profiling file
    * *out_units* - units used for output
    * *start_from* - plot events started after start_from
    * *stop_at* - plot events started before stop_at
    * *delta* - plot only events with duration >= delta
    * *only_averages* - outputs only the average timings
    """
    y_limits = set([])  # make plot well visible in vertical direction
    func_colors = {}
    averages = {}
    counts = {}
    max_func_name = 0
    events_infos = []

    print "Retreiving data..."

    if not only_averages:
        plt.figure()
    for attr in data:
        y_limits.add(attr)
        for event in data[attr]:
            start = q.Quantity(getattr(event, states[0]), file_units).\
                rescale(out_units)
            stop = q.Quantity(getattr(event, states[1]), file_units).\
                rescale(out_units)
            if start >= start_from and start <= stop_at and\
                    stop - start >= delta:
                if not only_averages:
                    events_infos.append((event.FUNC_NAME, stop-start,
                                         start, stop, out_units))
                    if event.FUNC_NAME in func_colors:
                        # assign color to the functions
                        plt.plot([start, stop], [attr, attr], linewidth=5.0,
                                 color=func_colors[event.FUNC_NAME])
                    if event.FUNC_NAME not in func_colors:
                        line = plt.plot([start, stop], [attr, attr],
                                        linewidth=5.0)[0]
                        func_colors[event.FUNC_NAME] = line.get_color()
                if max_func_name < len(event.FUNC_NAME):
                    max_func_name = len(event.FUNC_NAME)
                if event.FUNC_NAME in averages:
                    averages[event.FUNC_NAME] += stop-start
                    counts[event.FUNC_NAME] += 1
                else:
                    averages[event.FUNC_NAME] = stop-start
                    counts[event.FUNC_NAME] = 1

    if not only_averages:
        print "Functions and times spent in them:"
        for ev_info in events_infos:
            print "{0:>{1}} - duration: {2:10.5f} start: {3:10.5f} ".format(
                ev_info[0], max_func_name, float(ev_info[1].magnitude),
                float(ev_info[2].magnitude)) +\
                "stop: {0:10.5f} {1}".format(float(ev_info[3].magnitude),
                                             ev_info[4].symbol)
        print
    print "Plot information:"
    print "states:", states, "start from:", start_from, "stop at:", stop_at,\
        "minimum event duration:", delta, out_units.symbol
    print
    print "Timing statistics:"
    for fn in averages:
        print "{0:>{1}}: called {2:5} times, average time {3:10.5f} {4}".\
            format(fn, max_func_name, counts[fn],
                   float(averages[fn])/counts[fn], out_units.symbol)
    if not only_averages:
        print
        print "Legend:"
        for fn in func_colors:
            print "{0:>{1}}: {2}".format(
                fn, max_func_name, colors[func_colors[fn]])
        plt.ylim(min(y_limits)-0.5, max(y_limits)+0.5)
        plt.xlabel(out_units.symbol)
        plt.ylabel(attribute)
        plt.show()


if __name__ == '__main__':
    usage = "usage: %prog [options] profile_file"
    parser = OptionParser(usage=usage)
    parser.add_option("-a", "--attribute", metavar="ATTRIBUTE",
                      dest="attribute", default=ProfileReconstructor.
                      attributes[1],
                      help="Attribute for which the events" +
                      " will be plotted. Can be one of the following: %s" %
                      ProfileReconstructor.attributes[:3])
    parser.add_option("-s", "--start", metavar="START", type="float",
                      default=0.0, dest="start",
                      help="Plot only events which started after " +
                      "the timestamp given by this value, (default: %default)")
    parser.add_option("-p", "--stop", metavar="STOP", type="float",
                      default=sys.float_info.max, dest="stop",
                      help="Plot only events which started before " +
                      "the timestamp given by this value, (default: %default)")
    parser.add_option("-e", "--entry", metavar="ENTRY", default="START",
                      dest="entry", help="Event status which is considered " +
                      "an entry point for every plotted task. One of %s" %
                      str(ProfileReconstructor.cl_states) +
                      ", (default: %default)")
    parser.add_option("-x", "--exit", metavar="EXIT", default="END",
                      dest="exit", help="Event status which is considered " +
                      "an exit point for every plotted task. One of %s" %
                      str(ProfileReconstructor.cl_states) +
                      ", (default: %default)")
    parser.add_option("-u", "--units", metavar="UNITS", default="ms",
                      dest="units", help="Time units. One of %s" %
                      (ProfileReconstructor.str_to_qtime.keys()) +
                      ", (default: %default)")
    parser.add_option("-d", "--delta", metavar="DELTA", type="float",
                      default=0.0, dest="delta",
                      help="Minimum time duration of " +
                      "an event which will be plotted. " +
                      "In units specified by UNITS "
                      + "option, (default: %default)")
    parser.add_option("--only-averages", metavar="ONLY_AVERAGES",
                      dest="only_averages",
                      default=0, action="count",
                      help="Output only average timings.")

    opts, args = parser.parse_args()

    if args == []:
        parser.print_help()
        sys.exit(0)

    if not os.path.exists(args[0]):
        print >> sys.stderr, "File \"%s\" does not exist." % args[0]
        sys.exit(0)

    if opts.attribute.upper() not in ProfileReconstructor.attributes:
        print >> sys.stderr, "Attribute \"%s\" not from %s." %\
            (opts.attribute, ProfileReconstructor.attributes[:3])
        sys.exit(0)

    if opts.entry.upper() not in ProfileReconstructor.cl_states:
        print >> sys.stderr, "Entry level \"%s\" not from %s." %\
            (opts.entry, ProfileReconstructor.cl_states)
        sys.exit(0)

    if opts.exit.upper() not in ProfileReconstructor.cl_states:
        print >> sys.stderr, "Exit level \"%s\" not from %s." %\
            (opts.entry, ProfileReconstructor.cl_states)
        sys.exit(0)

    print "\n\n\n***** Warning *****"
    print "Relative device timings are only estimates!\n\n\n"

    # init OK, plot the data
    pr = ProfileReconstructor(args[0], opts.units)
    plot(pr.get_data(opts.attribute.upper()), opts.attribute.upper(),
         (opts.entry.upper(), opts.exit.upper()), pr.file_units, pr.units,
         opts.start, opts.stop, opts.delta, opts.only_averages > 0)
