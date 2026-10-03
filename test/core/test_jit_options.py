"""``NUMBOX_JIT_OPTIONS``, read once when ``numbox.core.configurations`` is imported, so read in a child."""
import os
import subprocess
import sys

import pytest

IMPORT_CONFIGURATIONS = "import numbox.core.configurations"


@pytest.mark.parametrize("value", ["null", "42", "[]", '"cache"'])
def test_a_value_that_is_not_an_object_is_refused_by_name(tmp_path, value):
    # Valid JSON of another shape went to njit as keyword arguments and died
    # there, or at the fallback's read of "cache", with a message naming
    # neither the variable nor the shape it takes.
    env = dict(os.environ, NUMBOX_JIT_OPTIONS=value)
    run = subprocess.run([sys.executable, "-c", IMPORT_CONFIGURATIONS], capture_output=True, text=True, env=env,
                         cwd=str(tmp_path))
    assert run.returncode != 0
    assert "NUMBOX_JIT_OPTIONS must be a JSON object" in run.stderr, run.stderr


@pytest.mark.parametrize("value", ['{"cache": "false"}', '{"cache": 0}', '{"cache": null}'])
def test_a_cache_value_that_is_not_a_boolean_is_refused_by_name(tmp_path, value):
    # The string "false", the natural quoting slip in an environment variable,
    # is true to numba, which reads the option's truth, so caching stayed on.
    env = dict(os.environ, NUMBOX_JIT_OPTIONS=value)
    run = subprocess.run([sys.executable, "-c", IMPORT_CONFIGURATIONS], capture_output=True, text=True, env=env,
                         cwd=str(tmp_path))
    assert run.returncode != 0
    assert 'NUMBOX_JIT_OPTIONS "cache" must be true or false' in run.stderr, run.stderr
