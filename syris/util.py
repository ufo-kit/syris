"""
Utility functions.
"""
import atexit
from optparse import OptionParser
import logging
from syris.profiling import profiler, Profiler

logger = logging.getLogger()

#=============================================================================
# Option parser
#=============================================================================

# Available for customization.
parser = OptionParser()

_initialized = False


def init(queues):
    global _initialized
    if _initialized:
        raise RuntimeError("Already initialized.")
    _initialized = True

    parser.add_option("-l", "--log", dest="logging_level",
                      help="logging level", metavar="LEVEL")
    parser.add_option("-p", "--profiler", dest="profiler", action="count",
                      default=0, help="enable profiling", metavar="PROFILE")
    parser.add_option("--profiler-file", dest="profiler_file", action="store",
                      default="profile.dat", help="profiler file",
                      metavar="PROFILER_FILE")
    parser.add_option("-o", "--logger-file", dest="logger_file",
                      default="simulation.log",
                      help="log file path", metavar="LOG_FILE")

    cmdoptions = parser.parse_args()[0]
    if cmdoptions.logging_level is not None:
        init_logging(logging.getLevelName(cmdoptions.logging_level),
                     cmdoptions.logger_file)

    if cmdoptions.profiler:
        profiler.init(cmdoptions.profiler_file, queues)
        profiler.start()

        @atexit.register
        def exit_handler():
            logger.info("Shutting down profiler...")
            profiler.shutdown()


def init_logging(level, logger_file):
    """Initialize logging with output to *logger_file*."""
    logger.setLevel(level)
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=fmt)

    fh = logging.FileHandler(logger_file, "w")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)
