"""Tests for ``numbox.utils.digest``, the hash of 16 hexadecimal digits that names generated code.

The tests check that the digest has 16 digits and is the same for the same inputs. They check that it changes with
the subject, with a function that differs in a literal and in its name, and with numbox's default jit options. It
also changes with the options of a dispatcher or a ``@vectorize`` function. A callable without code is hashed as a
cloudpickle that holds its state. A function that reads a value without a fingerprint is hashed with the best-effort
fingerprint, which still folds closure values. All the digests are compared within one process.
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
    # The two functions differ in co_consts and in their names; their bytecode is the same on Python 3.10 to 3.13. The
    # fingerprint folds both, so the test does not show which of the two changes the digest.
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


def test_digest_varies_with_a_named_function_that_a_module_attribute_holds():
    """The digest of a callback that calls ``helpers.op`` changes with the builtin function that the attribute holds.

    That holds for a Python function and for a dispatcher. With one digest for both, a SQLite registration in the next
    process would load the code that was compiled for the first function.
    """
    import math
    import types

    helpers = types.ModuleType("helpers")

    def callback(x):
        return helpers.op(x)

    for fn in (callback, njit(callback)):
        digests = []
        for op in (math.sin, math.cos):
            helpers.op = op
            digests.append(digest("subject", [fn]))
        assert digests[0] != digests[1]


def test_digest_of_a_callback_that_calls_a_helper_without_a_fingerprint_through_a_module_holds_the_data_it_reads():
    """A callback that calls ``helpers.step``, which has no fingerprint, keeps ``helpers.SCALE`` in its digest.

    numbox compiles a struct or a kernel without a cache when such a helper leaves a function without a fingerprint.
    A digest only names code, so a SQLite registration stays cached. The registration in the next process would load
    the code compiled for the first value of ``helpers.SCALE`` if both values gave one digest. The digest does not
    depend on ``helpers.UNUSED``, which the callback does not read. A cloudpickle of a ``@vectorize`` callback would
    hold ``helpers.UNUSED``.
    """
    import types

    from numba import float64

    helpers = types.ModuleType("helpers")

    @njit
    def step(x):
        return float64(x)  # A numba type read by name has no fingerprint, so step has none.

    def callback(x):
        return helpers.step(x) * helpers.SCALE

    helpers.step = step
    for fn in (callback, njit(callback), vectorize(callback)):
        digests = []
        for scale, unused in ((2.0, 1), (2.0, 2), (3.0, 2)):
            helpers.SCALE, helpers.UNUSED = scale, unused
            digests.append(digest("subject", [fn]))
        assert digests[0] == digests[1] != digests[2]


def test_digest_changes_with_a_jitted_function_in_a_tuple_that_calls_a_helper_without_a_fingerprint():
    """The digest of a callback that calls ``helpers.OPS[0]`` changes with the body of the function in the tuple.

    The function in the tuple calls ``low.raw``, which has no fingerprint, through a module. The callback then has no
    fingerprint of the kind that turns a cache off, and its digest folds the tuple as it did before that rule.
    """
    import types

    from numba import float64

    low = types.ModuleType("low")
    helpers = types.ModuleType("helpers")

    @njit
    def raw(x):
        return float64(x)  # A numba type read by name has no fingerprint, so raw has none.

    low.raw = raw

    def callback(x):
        return helpers.OPS[0](x)

    digests = []
    for factor in (2.0, 3.0):
        helpers.OPS = (njit(lambda x, factor=factor: factor * low.raw(x)),)
        digests.append(digest("subject", [callback]))
    assert digests[0] != digests[1]


if __name__ == '__main__':
    collect_and_run_tests(__name__)
