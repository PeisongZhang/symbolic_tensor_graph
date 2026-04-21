"""Regression test for correctness_todo.md §3: when activation_recompute=True,
`_print_gpu_vram` must NOT apply the old hard-coded 0.2 keep-ratio and must
emit a note that points to ASTRA-sim's `peak memory usage` as authoritative.

Run with:
    cd dnn_workload/symbolic_tensor_graph
    python3 test_cases/test_vram_ar_note.py
"""
import io
import os
import sys
from contextlib import redirect_stdout

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from symbolic_tensor_graph.vram_counting import _print_gpu_vram


class _FakeTensor:
    def __init__(self, name, require_grads=False, grad_of=None, y_shape=None):
        self.name = name
        self.id = name
        self.require_grads = require_grads
        self.grad_of = grad_of
        self.y_shape = y_shape or []


class _FakeGraph:
    def __init__(self, tensors):
        self.tensors = tensors


class _FakeBundle:
    def __init__(self, graphs):
        self.graphs = graphs


def _capture(fn):
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


def test_ar_on_does_not_emit_keep_ratio_token():
    # Empty graph is enough: we only care about the note text.
    bundle = _FakeBundle(
        graphs={(("dp", 0),): _FakeGraph(tensors=[])}
    )
    out = _capture(lambda: _print_gpu_vram(
        bundle, symbol_map={}, mixed_precision=False, header="[T] ",
        activation_recompute=True,
    ))
    assert "@0.2" not in out, "old hard-coded @0.2 keep-ratio still in output"
    assert "AR=on" in out and "peak memory usage" in out, (
        "AR-on note must mention 'peak memory usage' as authoritative source; "
        f"got: {out!r}"
    )


def test_ar_off_no_note():
    bundle = _FakeBundle(graphs={(("dp", 0),): _FakeGraph(tensors=[])})
    out = _capture(lambda: _print_gpu_vram(
        bundle, symbol_map={}, mixed_precision=False, header="[T] ",
        activation_recompute=False,
    ))
    assert "AR=on" not in out, "AR note leaked when AR=off"
    assert "@0.2" not in out, "hard-coded 0.2 leaked"


def test_keep_ratio_kwarg_removed():
    # The public signature should no longer accept activation_recompute_keep_ratio.
    import inspect
    sig = inspect.signature(_print_gpu_vram)
    assert "activation_recompute_keep_ratio" not in sig.parameters, (
        "activation_recompute_keep_ratio still present in signature; "
        "path A required removing it"
    )


if __name__ == "__main__":
    test_ar_on_does_not_emit_keep_ratio_token()
    test_ar_off_no_note()
    test_keep_ratio_kwarg_removed()
    print("test_vram_ar_note: OK (3/3)")
