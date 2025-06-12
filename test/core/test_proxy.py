import re
from numba import float64
from numba.core.types import Omitted
from numbox.core.proxy.proxy import proxy, make_proxy_name


aux_1_sig = float64(float64)


@proxy(aux_1_sig, jit_options={'cache': True})
def aux_1(x):
    return 3.14 * x


def test_1():
    assert abs(aux_1(2.2) - 3.14 * 2.2) < 1e-15
    llvm_ir = next(iter(aux_1.inspect_llvm().values()))
    assert aux_1.__name__ == make_proxy_name('aux_1')
    if '@cfunc.' in llvm_ir:
        cfunc_name = r"double @cfunc\.\w+aux_1\w+\(double"  # noqa: W605
        assert len(re.findall(f"declare {cfunc_name}", llvm_ir)) == 1
        assert len(re.findall(f"call {cfunc_name}", llvm_ir)) == 1
    else:
        print(f"LLVM inspection disabled for cached code, {aux_1.__name__}")


aux_2_sig = [float64(float64, float64), float64(float64, Omitted(1.3))]


@proxy(aux_2_sig, jit_options={'cache': True})
def aux_2(x, *, y=1.3):
    return 3.14 * x + y


def test_2():
    assert abs(aux_2(2.2) - (3.14 * 2.2 + 1.3)) < 1e-15
    assert abs(aux_2(2.2, 1.4) - (3.14 * 2.2 + 1.4)) < 1e-15


if __name__ == '__main__':
    test_1()
    test_2()
