"""Regression tests for P0-B: interleaved pipeline mapping.

Verifies that --pipeline_virtual_stages > 1 produces non-contiguous chunk
assignment (Megatron-LM interleaved), and that v=1 is backward-compatible
with the original contiguous mapping.

Run with:
    cd dnn_workload/symbolic_tensor_graph
    python3 -m pytest test_cases/test_pipeline_interleaved.py -v
or a plain script:
    python3 test_cases/test_pipeline_interleaved.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from main import _build_chunk_cumulative_bounds, _block_idx_to_device


def _mapping(num_stacks, pp, v):
    cum = _build_chunk_cumulative_bounds(num_stacks, v * pp)
    return [_block_idx_to_device(i, cum, pp) for i in range(num_stacks)]


def test_v1_contiguous_8_blocks_pp4():
    assert _mapping(8, 4, 1) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_v1_remainder_9_blocks_pp4():
    # Remainder (1 extra block) goes to the first chunk -> device 0.
    assert _mapping(9, 4, 1) == [0, 0, 0, 1, 1, 2, 2, 3, 3]


def test_v2_interleaved_8_blocks_pp4():
    # 8 chunks of size 1; chunk j goes to device j%pp.
    assert _mapping(8, 4, 2) == [0, 1, 2, 3, 0, 1, 2, 3]


def test_v2_interleaved_16_blocks_pp4():
    # 8 chunks of size 2; blocks 0,1 on dev 0; 2,3 on dev 1; ...; 8,9 on dev 0.
    assert _mapping(16, 4, 2) == [0, 0, 1, 1, 2, 2, 3, 3,
                                  0, 0, 1, 1, 2, 2, 3, 3]


def test_v4_interleaved_16_blocks_pp4():
    # 16 chunks of size 1; chunk j -> device j%4.
    assert _mapping(16, 4, 4) == [0, 1, 2, 3] * 4


def test_pp1_always_device_0():
    for v in (1, 2, 4):
        assert all(d == 0 for d in _mapping(8, 1, v))


def test_et_generation_interleaved_cross_device_more_p2p(tmp_path=None):
    """End-to-end: generate a tiny workload with v=1 and v=2, and verify that
    the interleaved variant produces strictly more SEND/RECV nodes (because
    adjacent blocks now live on different devices)."""
    import subprocess
    import tempfile

    from google.protobuf.internal.decoder import _DecodeVarint32
    _PB_PATH = os.path.abspath(os.path.join(
        _HERE, "..",
        "symbolic_tensor_graph/chakra/backends/chakra_00_4_backend/et_def",
    ))
    sys.path.insert(0, _PB_PATH)
    import et_def_pb2

    def count_p2p(path):
        with open(path, "rb") as f:
            data = f.read()
        off = 0
        ln, off = _DecodeVarint32(data, off)
        off += ln  # skip GlobalMetadata
        send = recv = 0
        while off < len(data):
            ln, off = _DecodeVarint32(data, off)
            n = et_def_pb2.Node()
            n.ParseFromString(data[off:off + ln])
            off += ln
            if n.type == et_def_pb2.COMM_SEND_NODE:
                send += 1
            elif n.type == et_def_pb2.COMM_RECV_NODE:
                recv += 1
        return send, recv

    def run(out_dir, v):
        cmd = [
            sys.executable, "main.py",
            "--output_dir", out_dir, "--output_name", "workload.%d.et",
            "--dp", "1", "--tp", "1", "--pp", "4", "--sp", "1",
            "--dvocal", "32000", "--dmodel", "512", "--dff", "1024",
            "--head", "8", "--kvhead", "8", "--num_stacks", "8",
            "--seq", "256", "--batch", "4", "--micro_batch", "1",
            "--model_type", "llama", "--mixed_precision", "true",
            "--pipeline_virtual_stages", str(v),
        ]
        subprocess.run(cmd, cwd=os.path.abspath(os.path.join(_HERE, "..")),
                       check=True, capture_output=True)

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        run(d1, 1)
        run(d2, 2)
        p2p_v1 = [count_p2p(os.path.join(d1, f"workload.{r}.et")) for r in range(4)]
        p2p_v2 = [count_p2p(os.path.join(d2, f"workload.{r}.et")) for r in range(4)]
        # Every rank should have strictly more communication under interleaved.
        for r, ((s1, r1), (s2, r2)) in enumerate(zip(p2p_v1, p2p_v2)):
            assert s2 > s1 and r2 > r1, (
                f"rank {r}: v1 send/recv = {s1}/{r1}, v2 send/recv = {s2}/{r2}"
            )


if __name__ == "__main__":
    # Simple driver — run all tests in order.
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"-- {name}")
            fn()
            print("   OK")
    print("ALL PASS")
