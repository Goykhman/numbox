import os
import json


def get_default_jit_options():
    """
    E.g., export NUMBOX_JIT_OPTIONS='{"cache": false}'
    """
    as_str = os.environ.get("NUMBOX_JIT_OPTIONS")
    if as_str is None:
        return {"cache": True}
    try:
        as_json = json.loads(as_str)
        return as_json
    except json.JSONDecodeError:
        raise ValueError("NUMBOX_JIT_OPTIONS must be valid JSON")


default_jit_options = get_default_jit_options()


MAX_STR_LENGTH = 2 ** 31 - 1
