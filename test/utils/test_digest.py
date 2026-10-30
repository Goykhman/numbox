"""Tests for ``numbox.utils.digest``, the hash of 16 hexadecimal digits that names generated code.

The tests check that the digest has 16 digits and is the same for the same inputs. They check that it changes with
the subject, with a literal in a function's body and with numbox's default jit options. It also changes with the
options of a dispatcher or a ``@vectorize`` function. A callable without code is hashed as a cloudpickle that holds
its state. A function that reads a value without a fingerprint is hashed with the best-effort fingerprint, which still
folds closure values. All the digests are compared within one process.
"""
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
    # The two bodies differ only in co_consts on Python 3.10 to 3.13, where their bytecode is the same. The fingerprint
    # folds co_consts.
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
    # A callable without code, such as a builtin, a functools.partial or a callable object, is hashed as a cloudpickle
    # of the object, and digest does not raise for it.
    assert len(digest("StateType", (functools.partial(len),))) == 16


def test_digest_codeless_callable_captures_state():
    # The cloudpickle holds the object's state, so two partials with different arguments have different digests. A
    # hash of the code alone would produce one digest for both.
    one = digest("StateType", (functools.partial(len, "a"),))
    two = digest("StateType", (functools.partial(len, "ab"),))
    assert one != two


def test_digest_empty_inputs_deterministic():
    assert digest("", ()) == digest("", ())
    assert len(digest("", ())) == 16


def test_digest_unfingerprintable_function_captures_closure():
    # cb reads marker, which has no fingerprint, so digest hashes cb with the best-effort fingerprint. That fingerprint
    # still folds closure values, so two functions with the same source and different values of k differ.
    class Opaque:  # an instance of this class has no fingerprint
        pass
    marker = Opaque()

    def make(k):
        def cb(x):
            return (marker, k, x)  # marker and k are both closure values of cb
        return cb

    assert digest("StateType", (make(1),)) != digest("StateType", (make(2),))


def test_digest_folds_dispatcher_jit_flags():
    # numba compiles different code for each set of jit options, so dispatchers of one body with different options must
    # have different digests. digest folds each dispatcher's Python function together with the options that the
    # dispatcher compiles under.
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
    # numba compiles a @vectorize function's scalar body under the function's options, so the options must change the
    # digest. A cloudpickle of the function holds the options too, so this test also passes if digest pickles a DUFunc,
    # unlike the dispatcher test above. digest folds a DUFunc through _canon_value instead, because a cloudpickle can
    # differ in every process for an object from the script that runs.
    def cb(x):
        return x + 1

    plain = digest("StateType", (vectorize(cb),))
    checked = digest("StateType", (vectorize(boundscheck=True)(cb),))
    assert plain != checked
    assert plain == digest("StateType", (vectorize(cb),))


if __name__ == '__main__':
    collect_and_run_tests(__name__)
