"""Wall-clock timing of function calls made from Python.

Decorating a function with ``timer`` records each call's duration in ``Timer.times`` under the function's name and
logs it at INFO on the ``numbox.utils.timer`` logger. Importing this module configures no logging, so the log records
do not appear until the application enables INFO for that logger, for instance with
``logging.basicConfig(level=logging.INFO)``.
"""
import logging
from time import perf_counter


logger = logging.getLogger(__name__)


class Timer:
    """A decorator that times each call of a function from Python and records the duration in ``Timer.times``.

    ``times`` is one dict that every ``Timer`` shares. It maps a function's ``__name__`` to the duration of its last
    call in seconds, so two timed functions with the same name share an entry. Each call is also logged at INFO, with
    the duration formatted to ``precision`` digits after the decimal point. A call that raises is neither recorded nor
    logged. The decorator returns a plain Python function named ``_``, which jitted code cannot call.
    """
    times = {}

    def __init__(self, precision=3):
        self.precision = precision

    def __call__(self, func):
        def _(*args, **kws):
            t_start = perf_counter()
            res = func(*args, **kws)
            t_end = perf_counter()
            duration = t_end - t_start
            logger.info(f"Execution of {func.__name__} took {duration:.{self.precision}f}s")
            self.times[func.__name__] = duration
            return res
        return _


timer = Timer()
