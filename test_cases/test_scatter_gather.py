"""Regression tests for P1-A: scatter/gather optimization.

Validates that enabling --scatter_gather_optimization:
  (a) reduces per-link cross-PP comm bytes by factor t (= tp degree);
  (b) inserts exactly one ALL_GATHER COLL_COMM node per cross-PP shadow on
      the receiver side.

Run with:
    cd dnn_workload/symbolic_tensor_graph
    python3 test_cases/test_scatter_gather.py
"""
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_STG_DIR = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(
    0,
    os.path.join(
        _STG_DIR,
        "symbolic_tensor_graph/chakra/backends/chakra_00_4_backend/et_def",
    ),
)
import et_def_pb2
from google.protobuf.internal.decoder import _DecodeVarint32


def _iter_nodes(path):
    with open(path, "rb") as f:
        data = f.read()
    off = 0
    ln, off = _DecodeVarint32(data, off)
    off += ln  # skip GlobalMetadata
    while off < len(data):
        ln, off = _DecodeVarint32(data, off)
        n = et_def_pb2.Node()
        n.ParseFromString(data[off:off + ln])
        off += ln
        yield n


def _attr_int(n, name):
    for a in n.attr:
        if a.name == name:
            return a.int64_val
    return None


def _stats(et_dir, num_ranks):
    totals = {"send_bytes": 0, "recv_bytes": 0, "send_n": 0, "recv_n": 0, "ag_n": 0}
    per_rank = []
    NT = et_def_pb2.NodeType
    for r in range(num_ranks):
        sb = rb = sn = rn = agn = 0
        for n in _iter_nodes(os.path.join(et_dir, f"workload.{r}.et")):
            if n.type == NT.COMM_SEND_NODE:
                sb += _attr_int(n, "comm_size") or 0
                sn += 1
            elif n.type == NT.COMM_RECV_NODE:
                rb += _attr_int(n, "comm_size") or 0
                rn += 1
            elif n.type == NT.COMM_COLL_NODE and "_Y_RECV_AG" in n.name:
                agn += 1
        per_rank.append((sb, rb, sn, rn, agn))
        totals["send_bytes"] += sb
        totals["recv_bytes"] += rb
        totals["send_n"] += sn
        totals["recv_n"] += rn
        totals["ag_n"] += agn
    return totals, per_rank


def _run(out_dir, sgo, tp, pp):
    cmd = [
        sys.executable, "main.py",
        "--output_dir", out_dir, "--output_name", "workload.%d.et",
        "--dp", "1", "--tp", str(tp), "--pp", str(pp), "--sp", "1",
        "--dvocal", "32000", "--dmodel", "512", "--dff", "1024",
        "--head", "8", "--kvhead", "8", "--num_stacks", "4",
        "--seq", "256", "--batch", "4", "--micro_batch", "1",
        "--model_type", "llama", "--mixed_precision", "true",
        "--scatter_gather_optimization", str(bool(sgo)).lower(),
    ]
    subprocess.run(cmd, cwd=_STG_DIR, check=True, capture_output=True)


def test_sg_reduces_p2p_bytes_by_factor_t():
    """tp=4, pp=2: per-link P2P bytes drop to 1/t; total ratio = 4."""
    tp, pp = 4, 2
    num_ranks = tp * pp
    with tempfile.TemporaryDirectory() as off, tempfile.TemporaryDirectory() as on:
        _run(off, False, tp, pp)
        _run(on, True, tp, pp)
        off_tot, _ = _stats(off, num_ranks)
        on_tot, on_per = _stats(on, num_ranks)

    # Same number of SEND/RECV nodes whether or not sg is on — only sizes change.
    assert off_tot["send_n"] == on_tot["send_n"], \
        f"SEND count changed: {off_tot['send_n']} vs {on_tot['send_n']}"
    assert off_tot["recv_n"] == on_tot["recv_n"], \
        f"RECV count changed: {off_tot['recv_n']} vs {on_tot['recv_n']}"

    # Bytes drop by factor t.
    assert off_tot["send_bytes"] == on_tot["send_bytes"] * tp, (
        f"expected off/on SEND bytes ratio = tp ({tp}); got "
        f"{off_tot['send_bytes']} / {on_tot['send_bytes']}"
    )
    assert off_tot["recv_bytes"] == on_tot["recv_bytes"] * tp, (
        f"expected off/on RECV bytes ratio = tp ({tp}); got "
        f"{off_tot['recv_bytes']} / {on_tot['recv_bytes']}"
    )

    # Each pp-boundary RECV on ON side spawns exactly one ALL_GATHER; totals
    # should match the number of RECV nodes.
    assert on_tot["ag_n"] == on_tot["recv_n"], (
        f"AG count != RECV count: {on_tot['ag_n']} vs {on_tot['recv_n']}"
    )
    assert off_tot["ag_n"] == 0


def test_sg_is_noop_when_tp_equals_1():
    """tp=1, pp=2: scatter/gather has no effect (each link is already minimal)."""
    tp, pp = 1, 4
    num_ranks = tp * pp
    with tempfile.TemporaryDirectory() as off, tempfile.TemporaryDirectory() as on:
        _run(off, False, tp, pp)
        _run(on, True, tp, pp)
        off_tot, _ = _stats(off, num_ranks)
        on_tot, _ = _stats(on, num_ranks)
    assert off_tot == on_tot, f"tp=1 SG should be a no-op; off={off_tot} on={on_tot}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"-- {name}")
            fn()
            print("   OK")
    print("ALL PASS")
