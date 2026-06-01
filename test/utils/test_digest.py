import functools

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


if __name__ == '__main__':
    collect_and_run_tests(__name__)
