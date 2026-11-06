"""Tests that an invalid ``NUMBOX_JIT_OPTIONS`` value fails the import with a message that names the variable.

The tests cover a value that is valid JSON but not an object, and an object whose ``"cache"`` is not ``true`` or
``false``. ``numbox.core.configurations`` reads the variable once, when it is first imported, so each test imports the
module in a child process with the variable set. Each test checks that the import fails and that the message shows the
expected form. The cache check that runs for valid options is tested in ``test_no_cache_location.py``.
"""
import os
import subprocess
import sys

import pytest

IMPORT_CONFIGURATIONS = "import numbox.core.configurations"


@pytest.mark.parametrize("value", ["null", "42", "[]", '"cache"'])
def test_a_value_that_is_not_an_object_is_refused_by_name(tmp_path, value):
    """A value that is valid JSON but not an object fails the import with a message that names the variable.

    numbox passes the object's keys to ``njit`` as keyword arguments, and its cache check reads ``"cache"`` from the
    object. ``get_jit_options`` refuses any other JSON type before either use, so the message names the variable and
    shows the expected form.
    """
    env = dict(os.environ, NUMBOX_JIT_OPTIONS=value)
    run = subprocess.run([sys.executable, "-c", IMPORT_CONFIGURATIONS], capture_output=True, text=True, env=env,
                         cwd=str(tmp_path))
    assert run.returncode != 0
    assert "NUMBOX_JIT_OPTIONS must be a JSON object" in run.stderr, run.stderr


@pytest.mark.parametrize("value", ['{"cache": "false"}', '{"cache": 0}', '{"cache": null}'])
def test_a_cache_value_that_is_not_a_boolean_is_refused_by_name(tmp_path, value):
    """A ``"cache"`` value other than ``true`` or ``false`` fails the import with a message that names the variable.

    numba tests the option for truth, so the string ``"false"``, a common quoting mistake in an environment variable,
    would leave caching enabled.
    """
    env = dict(os.environ, NUMBOX_JIT_OPTIONS=value)
    run = subprocess.run([sys.executable, "-c", IMPORT_CONFIGURATIONS], capture_output=True, text=True, env=env,
                         cwd=str(tmp_path))
    assert run.returncode != 0
    assert 'NUMBOX_JIT_OPTIONS "cache" must be true or false' in run.stderr, run.stderr
