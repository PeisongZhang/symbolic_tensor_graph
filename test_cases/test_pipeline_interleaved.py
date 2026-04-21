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

from symbolic_tensor_graph.graph.pipeline_schedule import (
    _apply_1f1b_interleaved_to_rank,
    _build_1f1b_interleaved_sequence,
    _classify_chunk_on_device,
    PHASE_F,
    PHASE_B,
)
from symbolic_tensor_graph.chakra.node import Node as _ChakraNode


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


def _make_fake_hybrid_graph(nodes):
    """Shape-compatible stand-in for HybridGraph for the pipeline-schedule
    post-process (it only iterates tensor_map_nodes.values())."""
    class _FakeHG:
        def __init__(self, nodes):
            self.tensor_map_nodes = {i: {"comp": n} for i, n in enumerate(nodes)}

    return _FakeHG(nodes)


def _make_comp_node(nid, name):
    n = _ChakraNode.__new__(_ChakraNode)
    n.id = nid
    n.name = name
    n.node_type = _ChakraNode.NodeType.COMP_NODE
    n.ctrl_deps = []
    n.data_deps = []
    return n


def test_interleaved_sequence_matches_megatron_rank0():
    # For rank=0, p=4, v=2, num_mb=8 the warmup is 10 F chunks.
    seq = _build_1f1b_interleaved_sequence(num_mb=8, p=4, v=2, rank=0)
    f_warmup = [entry for entry in seq[:10]]
    assert f_warmup == [
        (PHASE_F, 0, 0), (PHASE_F, 1, 0), (PHASE_F, 2, 0), (PHASE_F, 3, 0),
        (PHASE_F, 0, 1), (PHASE_F, 1, 1), (PHASE_F, 2, 1), (PHASE_F, 3, 1),
        (PHASE_F, 4, 0), (PHASE_F, 5, 0),
    ]
    # Every (mb, chunk) appears once in F and once in B across the sequence.
    f_pairs = [(mb, ch) for ph, mb, ch in seq if ph == PHASE_F]
    b_pairs = [(mb, ch) for ph, mb, ch in seq if ph == PHASE_B]
    expected = {(mb, ch) for mb in range(8) for ch in range(2)}
    assert set(f_pairs) == expected and len(f_pairs) == 16
    assert set(b_pairs) == expected and len(b_pairs) == 16


def test_classify_chunk_skips_shadows_and_maps_transformer_blocks():
    block_to_chunk_local = {0: 0, 1: 0, 2: 0, 3: 0,
                            4: 1, 5: 1, 6: 1, 7: 1}
    v = 2
    # transformer block goes to its chunk.
    assert _classify_chunk_on_device(
        "mb0.transformer.4.attn.x@forward_COMP", block_to_chunk_local, v,
    ) == 1
    # shadows are excluded.
    assert _classify_chunk_on_device(
        "shadow_mb0.transformer.3.y@forward_Y_RECV",
        block_to_chunk_local, v,
    ) is None
    # in_emb is first chunk, loss/out_emb are last chunk.
    assert _classify_chunk_on_device(
        "mb0.in_emb@forward_COMP", block_to_chunk_local, v,
    ) == 0
    assert _classify_chunk_on_device(
        "mb0.out_emb@forward_COMP", block_to_chunk_local, v,
    ) == v - 1
    assert _classify_chunk_on_device(
        "mb0.loss@forward_COMP", block_to_chunk_local, v,
    ) == v - 1


def test_apply_1f1b_interleaved_inserts_cross_chunk_ctrl_deps():
    """End-to-end on a synthetic graph: for p=4, v=2, m=8, rank=0, verify
    that adjacent (mb, phase, chunk) groups in _build_1f1b_interleaved_sequence
    receive a ctrl_dep between them, and that nothing in the partition lacks
    such a link on a chain boundary."""
    num_mb, p, v, rank = 8, 4, 2, 0
    # rank 0 owns blocks 0 (chunk 0) and 4 (chunk 1) under v=2, p=4, num_stacks=8.
    block_to_chunk_local = {0: 0, 1: 0, 2: 0, 3: 0,
                            4: 1, 5: 1, 6: 1, 7: 1}
    blocks_on_rank0 = [0, 4]

    # Build COMP nodes in the order they would be added: per-block F pass,
    # then per-block B pass, micro-batch-by-micro-batch. Node IDs ascend.
    nodes = []
    nid = 1
    for mb in range(num_mb):
        for block in blocks_on_rank0:
            nodes.append(_make_comp_node(
                nid, f"mb{mb}.transformer.{block}.attn.x@forward_COMP",
            ))
            nid += 1
        for block in blocks_on_rank0:
            nodes.append(_make_comp_node(
                nid, f"mb{mb}.transformer.{block}.attn.dx@backward_COMP",
            ))
            nid += 1

    # Index by (mb, phase, chunk).
    def phase_of(name):
        return PHASE_B if ".dx@" in name else PHASE_F
    def chunk_of(name):
        return _classify_chunk_on_device(name, block_to_chunk_local, v)
    def mb_of(name):
        return int(name[2:name.index(".")])
    groups = {}
    for n in nodes:
        key = (mb_of(n.name), phase_of(n.name), chunk_of(n.name))
        groups.setdefault(key, []).append(n)
    for lst in groups.values():
        lst.sort(key=lambda n: n.id)

    hg = _make_fake_hybrid_graph(nodes)
    _apply_1f1b_interleaved_to_rank(
        hg, num_mb, p, v, rank, block_to_chunk_local,
    )

    # Walk the expected sequence and assert ctrl_deps link adjacent groups.
    seq = _build_1f1b_interleaved_sequence(num_mb, p, v, rank)
    prev_last_id = None
    link_count = 0
    for phase, mb, chunk in seq:
        key = (mb, phase, chunk)
        assert key in groups, f"missing group {key}"
        first = groups[key][0]
        if prev_last_id is not None:
            assert prev_last_id in first.ctrl_deps, (
                f"missing ctrl_dep {prev_last_id} -> first of {key} "
                f"(id={first.id}); got {first.ctrl_deps}"
            )
            link_count += 1
        prev_last_id = groups[key][-1].id
    # Sanity: we expect len(seq) - 1 injected edges.
    assert link_count == len(seq) - 1


def test_apply_1f1b_interleaved_v1_falls_back():
    """With v=1, interleaved should degrade to plain 1F1B and not crash
    when block_to_chunk_local is empty or omitted."""
    nodes = [_make_comp_node(1, "mb0.transformer.0.attn.x@forward_COMP"),
             _make_comp_node(2, "mb1.transformer.0.attn.x@forward_COMP"),
             _make_comp_node(3, "mb0.transformer.0.attn.dx@backward_COMP"),
             _make_comp_node(4, "mb1.transformer.0.attn.dx@backward_COMP")]
    hg = _make_fake_hybrid_graph(nodes)
    _apply_1f1b_interleaved_to_rank(
        hg, num_mb=2, p=2, v=1, rank=0, block_to_chunk_local={0: 0},
    )
    # plain 1F1B on rank 0 with num_mb=2, p=2: warmup=2 F, steady=0, cooldown=2 B.
    # We just verify at least one ctrl_dep was added.
    all_ctrl = [cd for n in nodes for cd in n.ctrl_deps]
    assert len(all_ctrl) >= 1


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
