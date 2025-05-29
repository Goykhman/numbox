import logging
from time import perf_counter


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Timer:
    times = {}

    def __call__(self, func):
        def _(*args, **kws):
            t_start = perf_counter()
            res = func(*args, **kws)
            t_end = perf_counter()
            duration = t_end - t_start
            logger.info(f"Execution of {func.__name__} took {duration:.3f}s")
            self.times[func.__name__] = duration
            return res
        return _


timer = Timer()
