import numpy
from numba import njit, typeof
from numba.core.types import Array, DictType, float64, int16, int64, unicode_type, void
from numba.typed.typeddict import Dict

from numbox.core.any.any_type import AnyType
from numbox.core.configurations import default_jit_options
from numbox.core.work.loader_utils import load_array_row_into_dict, np_struct_member_type
from numbox.core.work.lowlevel_work_utils import ll_make_work
from numbox.utils.highlevel import cres
from test.auxiliary_utils import collect_and_run_tests


numpy.random.seed(137)
NUM_OF_ENTITIES = 10


data_ty = numpy.dtype([
    ("quantity", numpy.int16),
    ("value", numpy.float64),
    ("state", "|S10")
], align=True)


def prepare_input_data(num_of_entities=NUM_OF_ENTITIES):
    data_ = numpy.empty((num_of_entities,), dtype=data_ty)
    quantity_ = numpy.random.randint(1, 10, size=num_of_entities, dtype=numpy.int16)
    value_ = 10 * numpy.random.rand(num_of_entities)
    state_ = numpy.random.choice(["liquid", "gas", "solid", "plasma", "qgcond"], size=num_of_entities).astype("|S10")
    data_["quantity"] = quantity_
    data_["value"] = value_
    data_["state"] = state_
    return data_


@njit(void(Array(typeof(data_ty).dtype, 1, "C"), int64, DictType(unicode_type, AnyType)), **default_jit_options)
def aux_load_array_row_into_dict(data, row_ind, loader_dict):
    return load_array_row_into_dict(data, row_ind, loader_dict)


def test_load_array_row_into_dict(num_of_entities=NUM_OF_ENTITIES):
    loader_dict = Dict.empty(unicode_type, AnyType)
    data = prepare_input_data()
    for i in range(num_of_entities):
        aux_load_array_row_into_dict(data, i, loader_dict)
        for member_name in list(data_ty.fields.keys()):
            assert loader_dict[member_name].get_as(
                np_struct_member_type(data_ty, member_name)
            ) == data[i][member_name]


derive_total_sig = float64(*[np_struct_member_type(data_ty, name) for name in data_ty.names])


@cres(derive_total_sig, **default_jit_options)
def derive_total(quantity_, value_, state_):
    if state_ == b"liquid":
        prod_ = 0.1 * quantity_ * value_
    elif state_ == b"gas":
        prod_ = 0.2 * quantity_ * value_
    elif state_ == b"solid":
        prod_ = 0.3 * quantity_ * value_
    elif state_ == b"plasma":
        prod_ = 0.4 * quantity_ * value_
    else:
        prod_ = numpy.nan
    return prod_


state_ty = np_struct_member_type(data_ty, "state")


@njit(**default_jit_options)
def make_graph(derive_total_):
    quantity = ll_make_work("quantity", int16(0), (), None)
    value = ll_make_work("value", 0.0, (), None)
    state = ll_make_work("state", b"", (), None, data_ty_ref=state_ty)

    total = ll_make_work("total", 0.0, (quantity, value, state), derive_total_)
    return total


@njit(**default_jit_options)
def run_entity(total, loader_dict):
    total.load(loader_dict)
    total.calculate()
    return total.data


@njit(**default_jit_options)
def run(total, data, loader_dict, num_of_entities=NUM_OF_ENTITIES):
    total_data = numpy.empty((num_of_entities,), dtype=float64)
    for i in range(num_of_entities):
        load_array_row_into_dict(data, i, loader_dict)
        total_data[i] = run_entity(total, loader_dict)
    return total_data


def test_loader_utils_1():
    total = make_graph(derive_total)
    data = prepare_input_data()
    loader_dict = Dict.empty(unicode_type, AnyType)
    total_data = run(total, data, loader_dict)

    state = data["state"]
    prod = data["value"] * data["quantity"]
    ref_array = numpy.select(
        [state == b"liquid", state == b"gas", state == b"solid", state == b"plasma", state == b"qgcond"],
        [0.1 * prod, 0.2 * prod, 0.3 * prod, 0.4 * prod, numpy.nan]
    )

    assert numpy.allclose(total_data, ref_array, equal_nan=True)


if __name__ == "__main__":
    collect_and_run_tests(__name__)
