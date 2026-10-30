"""Tests for ``numbox.utils.timer``, the decorator that times calls made from Python.

The tests check that a timed function returns its result and records its duration in ``Timer.times``, and that each
call is logged at INFO on the ``numbox.utils.timer`` logger. A subprocess checks that importing the module adds no
handler to the root logger, so that the application decides whether the records appear.
"""
import logging
import subprocess
import sys

from numbox.utils.timer import Timer
from test.auxiliary_utils import collect_and_run_tests


def test_timer_records_duration_and_returns_result():
    t = Timer()

    def _timer_test_add(x, y):
        return x + y

    timed = t(_timer_test_add)
    assert timed(2, 3) == 5
    assert Timer.times["_timer_test_add"] >= 0.0


class _Records(logging.Handler):
    """A logging handler that keeps every record it receives in ``records``.

    The test uses it instead of pytest's ``caplog`` fixture, so the file also runs through its ``collect_and_run_tests``
    block, which calls each test without arguments.
    """

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def test_timer_logs_at_info():
    t = Timer(precision=2)

    def _timer_test_noop():
        return None

    timer_logger = logging.getLogger("numbox.utils.timer")
    handler = _Records()
    level = timer_logger.level
    timer_logger.addHandler(handler)
    timer_logger.setLevel(logging.INFO)
    try:
        t(_timer_test_noop)()
    finally:
        timer_logger.removeHandler(handler)
        timer_logger.setLevel(level)
    assert len(handler.records) == 1
    assert handler.records[0].levelno == logging.INFO
    assert handler.records[0].getMessage().startswith("Execution of _timer_test_noop took ")


def test_import_leaves_root_logger_unconfigured():
    # A new interpreter runs the import, so the handlers that pytest adds to the root logger in this process do not
    # count.
    code = (
        "import logging\n"
        "import numbox.utils.timer\n"
        "print(len(logging.getLogger().handlers))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "0"


if __name__ == '__main__':
    collect_and_run_tests(__name__)
