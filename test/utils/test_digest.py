import functools

from numba import njit, vectorize

from numbox.utils.digest import digest
from test.auxiliary_utils import collect_and_run_tests


def test_digest_is_deterministic():
    def cb(x):
        return x + 1
    assert digest("StateType", (cb,)) == digest("StateType", (cb,))


def test_digest_length_is_16():
    def cb(x):
        return x
    assert len(digest("StateType", (cb,))) == 16


def test_digest_varies_with_subject():
    def cb(x):
        return x
    assert digest("AType", (cb,)) != digest("BType", (cb,))


def test_digest_varies_with_function_literal():
    # co_consts-sensitive: bodies differing only by a numeric literal must produce
    # different digests (cloudpickle of __code__, not bare co_code which is identical here).
    def cb_one(x):
        return x + 1

    def cb_two(x):
        return x + 2

    assert digest("StateType", (cb_one,)) != digest("StateType", (cb_two,))


def test_digest_varies_with_jit_options(monkeypatch):
    from numbox.utils import digest as digest_mod

    def cb(x):
        return x
    before = digest("StateType", (cb,))
    monkeypatch.setitem(digest_mod.jit_options, "cache", not digest_mod.jit_options.get("cache", True))
    assert digest("StateType", (cb,)) != before


def test_digest_accepts_codeless_callable():
    # a callable with no __code__ (builtin, functools.partial, callable object)
    # must hash via cloudpickle of the object itself instead of raising.
    assert len(digest("StateType", (functools.partial(len),))) == 16


def test_digest_codeless_callable_captures_state():
    # object-fallback hashes the whole object, so differing partial args (state)
    # yield different digests -- which __call__.__code__ alone would miss.
    one = digest("StateType", (functools.partial(len, "a"),))
    two = digest("StateType", (functools.partial(len, "ab"),))
    assert one != two


def test_digest_empty_inputs_deterministic():
    assert digest("", ()) == digest("", ())
    assert len(digest("", ())) == 16


def test_digest_unfingerprintable_function_captures_closure():
    # When the strict walker aborts on an un-canonicalizable referenced value, the
    # fallback must still capture closure state: two functions with identical
    # source but different closure values yield different digests. The old
    # fallback cloudpickled the bare __code__, dropping exactly that state (H7).
    class Opaque:  # not a canonicalizable type -> forces the strict walker to abort
        pass
    marker = Opaque()

    def make(k):
        def cb(x):
            return (marker, k, x)  # references the opaque global; k lives in the closure
        return cb

    assert digest("StateType", (make(1),)) != digest("StateType", (make(2),))


def test_digest_folds_dispatcher_jit_flags():
    # A Dispatcher's py_func is only half its identity: numba compiles a
    # different library per jit-flag set, so two dispatchers over one body must
    # key distinctly. The py_func shortcut saw only the Python body and collided
    # all of these onto a single digest (M12).
    def cb(x):
        return x + 1

    variants = [
        njit(cb),
        njit(nogil=True)(cb),
        njit(fastmath=True)(cb),
        njit(error_model="numpy")(cb),
        njit(boundscheck=True)(cb),
    ]
    digests = [digest("StateType", (v,)) for v in variants]
    assert len(set(digests)) == len(digests)


def test_digest_of_dispatcher_is_deterministic_and_body_sensitive():
    def cb(x):
        return x + 1

    def other(x):
        return x + 2

    assert digest("StateType", (njit(cb),)) == digest("StateType", (njit(cb),))
    assert digest("StateType", (njit(cb),)) != digest("StateType", (njit(other),))


def test_digest_folds_dufunc_jit_flags():
    # @vectorize freezes its scalar body at compile time, so its targetoptions
    # belong in the key too. Note this holds on the cloudpickle path as well, so
    # unlike the Dispatcher case above it does not by itself pin the _canon_value
    # routing -- what routing buys DUFuncs is escaping cloudpickle's documented
    # nondeterminism for __main__ objects.
    def cb(x):
        return x + 1

    plain = digest("StateType", (vectorize(cb),))
    checked = digest("StateType", (vectorize(boundscheck=True)(cb),))
    assert plain != checked
    assert plain == digest("StateType", (vectorize(cb),))


if __name__ == '__main__':
    collect_and_run_tests(__name__)
