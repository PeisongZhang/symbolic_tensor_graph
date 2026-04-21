"""Pipeline schedule post-process (P0-A).

Injects synthetic ctrl_deps into a BundledHybridGraph so that the per-rank
execution follows a specific pipeline schedule. Intended to be called after
MicroBatchReplicator + BundledConvertChakra, before readout.

Supported schedules:
    - "natural"            : no changes; pure dependency-driven execution.
    - "gpipe"              : all forward passes of all micro-batches complete
                             before any backward pass begins (per rank).
    - "1f1b"               : Megatron-LM / PipeDream-Flush 1F1B ordering with
                             warmup / steady / cool-down phases (per rank).
    - "1f1b-interleaved"   : chunk-level 1F1B used with v>1 virtual stages
                             (Megatron interleaved). Requires `virtual_stages`.

Design notes:
- The post-process operates on HybridGraph.tensor_map_nodes directly, adding
  integer node IDs into `node.ctrl_deps`. It preserves all data deps and
  existing ctrl deps.
- F/B classification comes from the tensor-name suffix. In STG, backward-pass
  tensors have names starting with 'd' + lowercase letter (e.g. 'dx', 'dy',
  'dw', 'dattn'), while forward-pass tensors do not. Weight-update and grad
  aggregation nodes are classified as "U" and not scheduled.
- Micro-batch index is extracted from the "mb{i}." prefix applied by
  MicroBatchReplicator in grad_updater.py.
- Node IDs within a rank's graph are monotonically increasing in topological
  order of graph construction, so "min id in subset = earliest node, max id
  in subset = latest node" is a safe proxy for identifying schedule boundaries.
"""
import re
from collections import defaultdict

from ..chakra.node import Node


_MB_PREFIX_RE = re.compile(r"^mb(\d+)\.")
_TENSOR_NAME_RE = re.compile(r"\.([A-Za-z_][A-Za-z_0-9]*)@")
_TRANSFORMER_BLOCK_RE = re.compile(r"transformer\.(\d+)\.")
_SHADOW_PREFIX = "shadow_"


# Phase enumeration --------------------------------------------------------
PHASE_F = "F"
PHASE_B = "B"
PHASE_U = "U"


def _classify_phase(node_name):
    """Return 'F', 'B', or 'U' for a Chakra node name.

    'U' covers optimizer / gradient-aggregation / weight-update nodes which
    happen after the backward pass and are not scheduled by 1F1B.
    """
    # Grad aggregation / weight update / shadow RECV for grad sync.
    if "_assembled_grad" in node_name or "_sharded_weight" in node_name:
        return PHASE_U
    m = _TENSOR_NAME_RE.search(node_name)
    if m is None:
        return PHASE_U
    tname = m.group(1)
    if (
        len(tname) >= 2
        and tname[0] == "d"
        and tname[1].isalpha()
        and tname[1].islower()
        # 'dff' is the FFN hidden dim (not a tensor), but it's never a tensor
        # name here; guard defensively.
        and tname != "dff"
    ):
        return PHASE_B
    return PHASE_F


def _extract_mb_index(node_name):
    """Return the mb index embedded in the node name, or None if absent.

    MicroBatchReplicator (grad_updater.py) prefixes every replicated tensor
    with "mb{i}." so every per-mb node carries this. Nodes without an "mb{i}."
    prefix (e.g. weight placeholders shared across mbs) return None and are
    excluded from scheduling.
    """
    # shadow nodes look like "shadow_mb{i}.transformer..."
    stripped = node_name
    if stripped.startswith("shadow_"):
        stripped = stripped[len("shadow_"):]
    m = _MB_PREFIX_RE.match(stripped)
    if m is None:
        return None
    return int(m.group(1))


# Core utilities -----------------------------------------------------------
def _iter_nodes(hybrid_graph):
    """Yield all Node objects within a HybridGraph."""
    for nodes_this_tensor in hybrid_graph.tensor_map_nodes.values():
        for node in nodes_this_tensor.values():
            yield node


def _partition_nodes(hybrid_graph):
    """Group nodes by (mb_idx, phase).

    Returns:
        partitions: dict[(mb_idx, phase) -> list[Node]] sorted by node.id
    """
    partitions = defaultdict(list)
    for node in _iter_nodes(hybrid_graph):
        mb = _extract_mb_index(node.name)
        if mb is None:
            continue
        phase = _classify_phase(node.name)
        if phase == PHASE_U:
            continue
        partitions[(mb, phase)].append(node)
    for key, lst in partitions.items():
        lst.sort(key=lambda n: n.id)
    return partitions


def _boundary_node_ids(partitions):
    """For each (mb, phase), return (first_id, last_id).

    Returns dict[(mb, phase) -> (first_id, last_id)].
    """
    bounds = {}
    for key, nodes in partitions.items():
        if not nodes:
            continue
        bounds[key] = (nodes[0].id, nodes[-1].id)
    return bounds


def _get_node_by_id(hybrid_graph, node_id):
    """Linear scan to find a node by its id. OK for correctness; not a hot path."""
    for node in _iter_nodes(hybrid_graph):
        if node.id == node_id:
            return node
    return None


def _add_ctrl_edge(hybrid_graph, parent_id, child_id):
    """Add parent_id to child_node.ctrl_deps (dedup)."""
    child = _get_node_by_id(hybrid_graph, child_id)
    if child is None:
        return
    if parent_id == child_id:
        return
    if parent_id in child.ctrl_deps:
        return
    child.ctrl_deps.append(parent_id)


def _find_pp_rank(readable_rank, pp_dim_symbol):
    """Extract integer pp rank from a readable_rank tuple like
    (('pp', 3), ('cp', 0), ('tp', 0), ('dp', 0)).

    pp_dim_symbol may be a sympy Symbol or a string; we compare by str()."""
    key_s = str(pp_dim_symbol)
    for sym, rank in readable_rank:
        if str(sym) == key_s:
            return int(rank)
    return None


# Schedule implementations -------------------------------------------------
def _apply_gpipe_to_rank(hybrid_graph, num_mb):
    """All F of all mbs before any B (per rank).

    ctrl_deps inserted:
      * Serialize F: last-F(mb_i) -> first-F(mb_{i+1})
      * F/B barrier: last-F(mb_{m-1}) -> first-B(mb_0)
      * Serialize B: last-B(mb_i)  -> first-B(mb_{i+1})
    """
    parts = _partition_nodes(hybrid_graph)
    bounds = _boundary_node_ids(parts)

    # Serialize F.
    for i in range(num_mb - 1):
        k_prev = (i, PHASE_F)
        k_next = (i + 1, PHASE_F)
        if k_prev in bounds and k_next in bounds:
            _add_ctrl_edge(hybrid_graph, bounds[k_prev][1], bounds[k_next][0])

    # F/B barrier: last F of last mb -> first B of first mb.
    k_last_f = (num_mb - 1, PHASE_F)
    k_first_b = (0, PHASE_B)
    if k_last_f in bounds and k_first_b in bounds:
        _add_ctrl_edge(hybrid_graph, bounds[k_last_f][1], bounds[k_first_b][0])

    # Serialize B.
    for i in range(num_mb - 1):
        k_prev = (i, PHASE_B)
        k_next = (i + 1, PHASE_B)
        if k_prev in bounds and k_next in bounds:
            _add_ctrl_edge(hybrid_graph, bounds[k_prev][1], bounds[k_next][0])


def _build_1f1b_sequence(num_mb, p, rank):
    """Build the 1F1B action sequence for a given rank.

    Returns a list of (phase, mb_idx) tuples, in execution order.

    rank is 0..p-1 where rank 0 = first stage.
    Warmup = p - rank forwards (stage 0 does all m F first if p-rank >= m).
    Cooldown = p - rank backwards.
    Steady: interleaved 1F 1B.
    """
    warmup = min(p - rank, num_mb)
    steady = max(0, num_mb - warmup)

    seq = []
    # Warmup: F of mb 0..warmup-1
    for i in range(warmup):
        seq.append((PHASE_F, i))
    # Steady: alternating 1F 1B.
    # After warmup, the first-backward mb is 0 and first-forward is `warmup`.
    for i in range(steady):
        mb_f = warmup + i
        mb_b = i
        seq.append((PHASE_B, mb_b))
        seq.append((PHASE_F, mb_f))
    # Cool-down: remaining backward passes.
    # Backward side has run `steady` already (indices 0..steady-1); still need
    # num_mb - steady backwards, indexed [steady, num_mb).
    for i in range(steady, num_mb):
        seq.append((PHASE_B, i))
    return seq


def _apply_1f1b_to_rank(hybrid_graph, num_mb, p, rank):
    """Serialize the rank's execution according to the 1F1B sequence."""
    seq = _build_1f1b_sequence(num_mb, p, rank)
    parts = _partition_nodes(hybrid_graph)
    bounds = _boundary_node_ids(parts)

    prev = None
    for phase, mb in seq:
        key = (mb, phase)
        if key not in bounds:
            # Phase not present on this rank (e.g. stage 0 has no B for
            # embedding). Skip without chaining to avoid stranded nodes.
            continue
        first_id, last_id = bounds[key]
        if prev is not None:
            _add_ctrl_edge(hybrid_graph, prev, first_id)
        prev = last_id


def _build_1f1b_interleaved_sequence(num_mb, p, v, rank):
    """Build 1F1B-interleaved action sequence per Megatron-LM.

    Follows the convention used by
    ``megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_with_interleaving``:

        microbatch_id_in_group  = k % (p * v)
        model_chunk_id          = microbatch_id_in_group // p
        microbatch_id           = (k // (p * v)) * p + (k % p)

    Backward chunks traverse chunks in reverse (``v - 1 - chunk``) while the
    microbatch mapping stays the same.

    Warmup (before the first backward) is
        (p - rank - 1) * 2 + (v - 1) * p
    clipped to ``v * num_mb`` (the total number of forward chunks on this
    device).

    Requires ``num_mb % p == 0`` so that chunk/mb indices stay in range; this
    mirrors Megatron's own requirement.

    Returns list of ``(phase, mb_idx, chunk_on_device)`` triples.
    """
    assert v >= 1 and p >= 1 and num_mb >= 1
    if v > 1 and num_mb % p != 0:
        raise ValueError(
            f"1f1b-interleaved requires num_micro_batches ({num_mb}) to be a "
            f"multiple of pipeline_parallel_size ({p})"
        )
    total_steps = v * num_mb
    warmup = (p - rank - 1) * 2 + (v - 1) * p
    warmup = max(0, min(total_steps, warmup))
    steady = total_steps - warmup

    def f_mb_chunk(k):
        mb = (k // (p * v)) * p + (k % p)
        chunk = (k % (p * v)) // p
        return mb, chunk

    def b_mb_chunk(k):
        mb, ch = f_mb_chunk(k)
        return mb, v - 1 - ch

    seq = []
    for k in range(warmup):
        mb, ch = f_mb_chunk(k)
        seq.append((PHASE_F, mb, ch))
    # Steady phase: 1 F of the next mb, then 1 B of an earlier mb.
    # On rank p-1 the first B in steady is B(mb=0, chunk=v-1); the matching
    # F(mb=0, chunk=v-1) only runs at steady k=0's F slot, so F must precede
    # B or the last rank deadlocks (B fires before its own F ever ran).
    for k in range(steady):
        mb_f, ch_f = f_mb_chunk(warmup + k)
        mb_b, ch_b = b_mb_chunk(k)
        seq.append((PHASE_F, mb_f, ch_f))
        seq.append((PHASE_B, mb_b, ch_b))
    for k in range(steady, total_steps):
        mb_b, ch_b = b_mb_chunk(k)
        seq.append((PHASE_B, mb_b, ch_b))
    return seq


def _classify_chunk_on_device(node_name, block_to_chunk_local, virtual_stages):
    """Return the chunk-on-device (``0..v-1``) the node belongs to, or None.

    Classification rules:
      - transformer.{N}.*  -> ``block_to_chunk_local[N]``
      - in_emb             -> 0 (first chunk on device 0)
      - out_emb / loss     -> ``v - 1`` (last chunk on device p-1)
      - shadow_*           -> None (these flow via data_deps from their
                                     producer on another rank; classifying
                                     them would pin the receive to the source
                                     rank's chunk which is meaningless here)
    """
    if node_name.startswith(_SHADOW_PREFIX):
        return None
    m = _TRANSFORMER_BLOCK_RE.search(node_name)
    if m is not None:
        block_idx = int(m.group(1))
        return block_to_chunk_local.get(block_idx)
    if "in_emb" in node_name:
        return 0
    if "out_emb" in node_name or "loss" in node_name:
        return virtual_stages - 1
    return None


def _partition_nodes_interleaved(hybrid_graph, block_to_chunk_local,
                                 virtual_stages):
    """Group COMP nodes by ``(mb_idx, phase, chunk_on_device)``.

    Only COMP nodes participate in the partition: SEND/RECV/coll-comm nodes
    are ordered transitively through their data_deps on the COMP nodes, and
    we want them free to overlap with compute on adjacent chunks rather than
    being pinned to the chunk they happen to be named after.
    """
    partitions = defaultdict(list)
    for node in _iter_nodes(hybrid_graph):
        if node.node_type != Node.NodeType.COMP_NODE:
            continue
        mb = _extract_mb_index(node.name)
        if mb is None:
            continue
        phase = _classify_phase(node.name)
        if phase == PHASE_U:
            continue
        chunk = _classify_chunk_on_device(
            node.name, block_to_chunk_local, virtual_stages,
        )
        if chunk is None:
            continue
        partitions[(mb, phase, chunk)].append(node)
    for lst in partitions.values():
        lst.sort(key=lambda n: n.id)
    return partitions


def _apply_1f1b_interleaved_to_rank(hybrid_graph, num_mb, p, v, rank,
                                    block_to_chunk_local):
    """Chunk-granularity 1F1B schedule injection.

    Requires ``block_to_chunk_local`` — a ``dict[block_idx -> chunk_on_device]``
    — so that COMP nodes can be partitioned by ``(mb, phase, chunk_on_device)``.
    Falls back to plain 1F1B at mb granularity when ``v <= 1`` or the metadata
    is missing (mirrors the prior behavior for non-interleaved usage).
    """
    if v <= 1 or not block_to_chunk_local:
        return _apply_1f1b_to_rank(hybrid_graph, num_mb, p, rank)

    seq = _build_1f1b_interleaved_sequence(num_mb, p, v, rank)
    parts = _partition_nodes_interleaved(
        hybrid_graph, block_to_chunk_local, v,
    )
    bounds = _boundary_node_ids(parts)

    prev = None
    for phase, mb, chunk in seq:
        key = (mb, phase, chunk)
        if key not in bounds:
            continue
        first_id, last_id = bounds[key]
        if prev is not None:
            _add_ctrl_edge(hybrid_graph, prev, first_id)
        prev = last_id


# Public API ---------------------------------------------------------------
class PipelineScheduleInjector:
    VALID_SCHEDULES = ("natural", "gpipe", "1f1b", "1f1b-interleaved")

    @classmethod
    def apply(
        cls,
        bundled_graph,
        schedule,
        num_micro_batches,
        pipeline_parallel_size,
        virtual_stages=1,
        pp_dim_symbol=None,
        block_to_chunk_local=None,
    ):
        """Inject ctrl_deps into each rank's HybridGraph per `schedule`.

        Args:
            bundled_graph: BundledHybridGraph from BundledConvertChakra.
            schedule: one of VALID_SCHEDULES.
            num_micro_batches: m, number of micro-batches per pipeline.
            pipeline_parallel_size: p.
            virtual_stages: v (>=1). Used only for 1f1b-interleaved.
            pp_dim_symbol: sympy symbol or string that identifies the pipeline
                dimension inside readable_rank tuples. Defaults to string 'pp'.
            block_to_chunk_local: dict[block_idx -> chunk_on_device]. Required
                for '1f1b-interleaved' with virtual_stages > 1; produced by
                main._create_pipeline_tensor_map[_mix_precision].

        Returns bundled_graph (modified in place).
        """
        if schedule not in cls.VALID_SCHEDULES:
            raise ValueError(
                f"Unknown pipeline schedule '{schedule}'. "
                f"Valid: {cls.VALID_SCHEDULES}"
            )
        if schedule == "natural":
            return bundled_graph

        if pp_dim_symbol is None:
            pp_dim_symbol = "pp"

        print(
            f"Applying pipeline schedule '{schedule}' "
            f"(p={pipeline_parallel_size}, m={num_micro_batches}, v={virtual_stages})"
        )
        for readable_rank, hybrid_graph in bundled_graph.graphs.items():
            rank = _find_pp_rank(readable_rank, pp_dim_symbol)
            if rank is None:
                continue
            if schedule == "gpipe":
                _apply_gpipe_to_rank(hybrid_graph, num_micro_batches)
            elif schedule == "1f1b":
                _apply_1f1b_to_rank(
                    hybrid_graph, num_micro_batches,
                    pipeline_parallel_size, rank,
                )
            elif schedule == "1f1b-interleaved":
                _apply_1f1b_interleaved_to_rank(
                    hybrid_graph, num_micro_batches,
                    pipeline_parallel_size, virtual_stages, rank,
                    block_to_chunk_local,
                )
        print("Pipeline schedule applied")
        return bundled_graph
