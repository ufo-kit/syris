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
import re
import sys
from matplotlib import pyplot as plt
from optparse import OptionParser
import os

LOGGER = logging.getLogger(__name__)

# Singleton.
PROFILER = None


class DummyProfiler():

    """A profiler which does nothing for saving time."""

    def __init__(self):
        pass

    def add(self, event, func_name=""):
        """Does nothing with input arguments."""
        pass


class Profiler(Thread):

    """An OpenCL GPU code PROFILER."""
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
        """Estimate the time shift between devices with respect to a
        global clock. This is important for evaluating relative device
        runtimes with respect to each other.
        """
        # Get only a single command queue for a device on which we will
        # determine the zero time of a device.
        unique_queues = []
        devices = []
        for queue in queues:
            if queue.device not in devices:
                unique_queues.append(queue)
                devices.append(queue.device)

        starts = {}
        start = time.time()
        for i in range(len(devices)):
            starts[devices[i]] = cl.enqueue_marker(unique_queues[i])
        d_t = (time.time() - start) * q.s

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
                                 d_t.rescale(q.ns))
        self._profile_file.write("# END_INIT\n")

    def run(self):
        """Run in a separate thread and serve the incoming events."""
        while not self.finish or not self._events.empty():
            try:
                event, kernel = self._events.get(timeout=0.1)
                cl.wait_for_events([event])
                self._process(event, kernel)
                self._events.task_done()
            except Empty:
                pass
            except Exception as exc:
                LOGGER.error(exc)
        self._profile_file.close()

    def shutdown(self):
        """Wait for all events to finish and then stop the PROFILER loop."""
        self.finish = True
        self.join()
        LOGGER.debug("Profiler finished.")

    def add(self, event, func_name=""):
        """Add an OpenCL *event* and function with name *func_name*
        into the PROFILER's queue.
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
        """Process an *event* from function with name *func_name*
        and write in into the profile file.
        """
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
        string = "%s\n%s\n%s\n%s\n" % (strings[0], strings[1], strings[2],
                                       strings[3])

        self._profile_file.write(string)


#=============================================================================
# Profiling Reconstruction
#=============================================================================


COLORS = {
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

    """A record in a profile file."""

    def __init__(self, *args):
        for att, val in args:
            setattr(self, att, val)

    def __str__(self):
        return "Record(" + str(self.__dict__) + ")"


class _Event(object):

    """An OpenCL event representation."""

    def __str__(self):
        return "Event(" + str(self.__dict__) + ")"


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
        of result dictionary. Return a dictionary in form
        {attr: {event_id: Event}}.
        """
        if attr == self._cache[1]:
            # last request was on the same attribute -> data do not change
            return self._cache[0]

        if self._events == {}:
            self._process()
        self._cache = self._get_aggregate(attr), attr

        return self._cache[0]

    def _analyze_line(self, string):
        """Analyze one line from the profile file."""
        match = ProfileReconstructor.pattern.search(string)

        vals = [int(match.group(ProfileReconstructor.attributes[0])),
                int(match.group(ProfileReconstructor.attributes[1])),
                int(match.group(ProfileReconstructor.attributes[2])),
                match.group(ProfileReconstructor.attributes[3]),
                match.group(ProfileReconstructor.attributes[4]),
                int(match.group(ProfileReconstructor.attributes[5]))
                ]
        args = zip(ProfileReconstructor.attributes, vals)

        return _Record(*args)

    def _process_header(self, file_obj):
        """Get the header information."""
        self._process_units(file_obj)
        self._process_t0(file_obj)
        self._process_dt_error(file_obj)

    def _process_units(self, file_obj):
        """Get information about units used in the profile file."""
        line = file_obj.readline()
        if line != "# units\n":
            raise ValueError("File corrupted.")
        line = file_obj.readline().strip()
        self.file_units = ProfileReconstructor.str_to_qtime[line.lower()]

    def _process_t0(self, file_obj):
        """Different devices have different time offsets,
        make the calibration.
        """
        line = ""
        while line != "# END_INIT_T0\n":
            line = file_obj.readline()
            if not line.startswith("#"):
                device_id, device_start = line.strip().split("\t")
                self._starts[int(device_id)] = int(device_start)

    def _process_dt_error(self, file_obj):
        """Error estimate for different device zero times with respect
        to global zero.
        """
        line = ""
        while line != "# END_INIT\n":
            line = file_obj.readline()
            if not line.startswith("#"):
                d_t = float(line.strip())
                print "Relative device time error: %g %s" % \
                    (q.Quantity(d_t, self.file_units).rescale(self.units),
                     self.units.symbol)

    def _process(self):
        """Process the whole profile file."""
        records = {}
        with open(self._profile_file_name, "r") as file_obj:
            self._process_header(file_obj)
            # skip the first one (explanation line)
            line = file_obj.readline()
            while True:
                line = file_obj.readline()
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
        """Get aggregated dictionary from an attribute."""
        prop = {}
        for event_id in self._events:
            if getattr(self._events[event_id], attribute) not in prop:
                prop[getattr(self._events[event_id], attribute)] = []
            prop[getattr(self._events[event_id], attribute)].\
                append(self._events[event_id])

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
                    events_infos.append((event.FUNC_NAME, stop - start,
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
                    averages[event.FUNC_NAME] += stop - start
                    counts[event.FUNC_NAME] += 1
                else:
                    averages[event.FUNC_NAME] = stop - start
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
    print "states:", states, "start from:", start_from, "stop at:", stop_at, \
        "minimum event duration:", delta, out_units.symbol
    print
    print "Timing statistics:"
    for f_name in averages:
        print "{0:>{1}}: called {2:5} times, average time {3:10.5f} {4}".\
            format(f_name, max_func_name, counts[f_name],
                   float(averages[f_name]) / counts[f_name], out_units.symbol)
    if not only_averages:
        print
        print "Legend:"
        for f_name in func_colors:
            print "{0:>{1}}: {2}".format(
                f_name, max_func_name, COLORS[func_colors[f_name]])
        plt.ylim(min(y_limits) - 0.5, max(y_limits) + 0.5)
        plt.xlabel(out_units.symbol)
        plt.ylabel(attribute)
        plt.show()


if __name__ == '__main__':
    USAGE = "usage: %prog [options] profile_file"
    PARSER = OptionParser(usage=USAGE)
    PARSER.add_option("-a", "--attribute", metavar="ATTRIBUTE",
                      dest="attribute", default=ProfileReconstructor.
                      attributes[1],
                      help="Attribute for which the events" +
                      " will be plotted. Can be one of the following: %s" %
                      ProfileReconstructor.attributes[:3])
    PARSER.add_option("-s", "--start", metavar="START", type="float",
                      default=0.0, dest="start",
                      help="Plot only events which started after " +
                      "the timestamp given by this value, (default: %default)")
    PARSER.add_option("-p", "--stop", metavar="STOP", type="float",
                      default=sys.float_info.max, dest="stop",
                      help="Plot only events which started before " +
                      "the timestamp given by this value, (default: %default)")
    PARSER.add_option("-e", "--entry", metavar="ENTRY", default="START",
                      dest="entry", help="Event status which is considered " +
                      "an entry point for every plotted task. One of %s" %
                      str(ProfileReconstructor.cl_states) +
                      ", (default: %default)")
    PARSER.add_option("-x", "--exit", metavar="EXIT", default="END",
                      dest="exit", help="Event status which is considered " +
                      "an exit point for every plotted task. One of %s" %
                      str(ProfileReconstructor.cl_states) +
                      ", (default: %default)")
    PARSER.add_option("-u", "--units", metavar="UNITS", default="ms",
                      dest="units", help="Time units. One of %s" %
                      (ProfileReconstructor.str_to_qtime.keys()) +
                      ", (default: %default)")
    PARSER.add_option("-d", "--delta", metavar="DELTA", type="float",
                      default=0.0, dest="delta",
                      help="Minimum time duration of " +
                      "an event which will be plotted. " +
                      "In units specified by UNITS "
                      + "option, (default: %default)")
    PARSER.add_option("--only-averages", metavar="ONLY_AVERAGES",
                      dest="only_averages",
                      default=0, action="count",
                      help="Output only average timings.")

    OPTS, ARGS = PARSER.parse_args()

    if ARGS == []:
        PARSER.print_help()
        sys.exit(0)

    if not os.path.exists(ARGS[0]):
        print >> sys.stderr, "File \"%s\" does not exist." % ARGS[0]
        sys.exit(0)

    if OPTS.attribute.upper() not in ProfileReconstructor.attributes:
        print >> sys.stderr, "Attribute \"%s\" not from %s." % \
            (OPTS.attribute, ProfileReconstructor.attributes[:3])
        sys.exit(0)

    if OPTS.entry.upper() not in ProfileReconstructor.cl_states:
        print >> sys.stderr, "Entry level \"%s\" not from %s." % \
            (OPTS.entry, ProfileReconstructor.cl_states)
        sys.exit(0)

    if OPTS.exit.upper() not in ProfileReconstructor.cl_states:
        print >> sys.stderr, "Exit level \"%s\" not from %s." % \
            (OPTS.entry, ProfileReconstructor.cl_states)
        sys.exit(0)

    print "\n\n\n***** Warning *****"
    print "Relative device timings are only estimates!\n\n\n"

    # init OK, plot the data
    PROF_RECO = ProfileReconstructor(ARGS[0], OPTS.units)
    plot(PROF_RECO.get_data(OPTS.attribute.upper()), OPTS.attribute.upper(),
         (OPTS.entry.upper(), OPTS.exit.upper()), PROF_RECO.file_units,
         PROF_RECO.units, OPTS.start, OPTS.stop, OPTS.delta,
         OPTS.only_averages > 0)
