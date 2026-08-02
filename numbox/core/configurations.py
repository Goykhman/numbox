import os
import json

from importlib.metadata import version


def get_jit_options():
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


jit_options = get_jit_options()


_PROXY_CACHE_STRICT_ENV = "NUMBOX_PROXY_CACHE_STRICT"


def _strict_cache_mode():
    """True when ``NUMBOX_PROXY_CACHE_STRICT`` selects strict validation of ``@proxy`` cache loads.

    Strict mode makes every fail-open path in the ``numbox.core.proxy.proxy`` cache guard loud: a payload
    that cannot be read, or an object whose container cannot be parsed, aborts the load with
    ``UnvalidatedProxyCacheError`` instead of loading unchecked; and a detected stale alias raises
    ``StaleProxyCacheError`` before the heal, leaving the stale entry on disk. It is a debugging aid, off by
    default. Unlike ``jit_options`` above, the value is read on each call rather than once at import, so the
    knob can be toggled within a process; the cost is one environment lookup against a multi-millisecond
    cache load. Anything other than the unset/empty/``0``/``false``/``no``/``off`` set (case-insensitive)
    enables it.
    """
    value = os.environ.get(_PROXY_CACHE_STRICT_ENV)
    return value is not None and value.strip().lower() not in ("", "0", "false", "no", "off")


MAX_STR_LENGTH = 2 ** 31 - 1


numba_version = int(version("numba").split(".")[1])
assert numba_version >= 60, numba_version

#: See the `FunctionModel` struct: (c_addr, py_addr[, jit_addr]). The third slot
#: arrived in numba 0.61.
#:
#: Kept here rather than beside its users in `numbox.utils.lowlevel`, because
#: importing that module compiles a cached `@proxy`, and both this constant and
#: `numbox.core.work.derive_wap` are needed on paths that must stay free of
#: compilation side effects.
function_struct_size = 3 if numba_version >= 61 else 2
