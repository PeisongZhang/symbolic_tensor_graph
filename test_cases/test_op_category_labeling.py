"""Regression test for correctness_todo.md §4: Chakra004Backend must stamp
every COMP_NODE with an `op_category` int attr mapped from STG op_type.

Run with:
    cd dnn_workload/symbolic_tensor_graph
    python3 test_cases/test_op_category_labeling.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend.chakra_00_4_backend import (
    Chakra004Backend,
)
from symbolic_tensor_graph.chakra.backends.backend import NodeBackendBase as _Base
from symbolic_tensor_graph.chakra.node import Node as FrontendNode


def _make_comp_node(op_type):
    n = FrontendNode()
    n.id = 1
    n.name = f"tensor_{op_type}_COMP"
    n.node_type = FrontendNode.NodeType.COMP_NODE
    n.num_ops = 1000
    n.tensor_size = 100
    n.y_tensor_size = 100
    n.op_type = op_type
    n.data_deps = []
    n.ctrl_deps = []
    return n


def _category_of_readout(op_type):
    n = _make_comp_node(op_type)
    backend_node = Chakra004Backend.readout(n)
    for attr in backend_node.attr:
        if attr.name == "op_category":
            return attr.int32_val
    return None


def test_gemm_maps_to_zero():
    assert _category_of_readout("M") == Chakra004Backend.OP_CATEGORY_GEMM == 0


def test_elemwise_family():
    for t in ("A", "E", "E2", "C"):
        got = _category_of_readout(t)
        assert got == Chakra004Backend.OP_CATEGORY_ELEMWISE == 1, (
            f"op_type={t} expected ELEMWISE=1, got {got}"
        )


def test_custom_maps_to_softmax():
    assert _category_of_readout("CUSTOM") == Chakra004Backend.OP_CATEGORY_SOFTMAX == 2


def test_broadcast_reduce_is_reduce():
    assert _category_of_readout("B") == Chakra004Backend.OP_CATEGORY_REDUCE == 3


def test_placeholder_is_other():
    # PlaceHolder / Identity / Reshape / Shadow / Slice → OTHER
    for t in ("T", "I", "R", "S", "SLICE"):
        got = _category_of_readout(t)
        assert got == Chakra004Backend.OP_CATEGORY_OTHER == 4, (
            f"op_type={t} expected OTHER=4, got {got}"
        )


def test_unknown_op_type_defaults_to_other():
    # Any string not in the table should fall back to OTHER rather than
    # silently emit a garbage category.
    assert (
        _category_of_readout("SOMETHING_UNEXPECTED")
        == Chakra004Backend.OP_CATEGORY_OTHER
    )


def test_attr_emitted_as_int32():
    n = _make_comp_node("M")
    backend_node = Chakra004Backend.readout(n)
    found = [a for a in backend_node.attr if a.name == "op_category"]
    assert len(found) == 1, "op_category attr must be emitted exactly once"
    # int32_val is the proto field we set; verify the message carries it.
    assert found[0].WhichOneof("value") == "int32_val", (
        "op_category must be written as int32_val (matches Roofline reader)"
    )


if __name__ == "__main__":
    test_gemm_maps_to_zero()
    test_elemwise_family()
    test_custom_maps_to_softmax()
    test_broadcast_reduce_is_reduce()
    test_placeholder_is_other()
    test_unknown_op_type_defaults_to_other()
    test_attr_emitted_as_int32()
    print("test_op_category_labeling: OK (7/7)")
