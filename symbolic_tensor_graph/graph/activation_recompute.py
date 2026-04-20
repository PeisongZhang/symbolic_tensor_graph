"""Activation recomputation post-process (P1-B).

Models Megatron-LM §3.5 selective activation recomputation by inflating each
transformer block's backward compute to cover the extra forward pass that
happens just before the backward pass.

Implementation strategy: for each (micro-batch, transformer-block) group on
each rank, compute
    F_total = sum of forward COMP nodes' num_ops
    B_total = sum of backward COMP nodes' num_ops
and scale every backward COMP node's num_ops by (1 + F_total / B_total).
Total per-block FLOP becomes F + (B + F) = B + 2F, matching the paper's
accounting (one extra forward before backward).

This is a coarse but self-consistent approximation: the extra FLOP is
serialized before backward via backward's existing data-dep chain (backward
already waits on forward activations), so the on-critical-path effect of
recompute is captured in wall-clock simulation. Peak VRAM modeling is
orthogonal and covered by P3-A (`LocalMemUsageTracker`).

Also updates `tensor_size` proportionally so that roofline operational
intensity stays consistent with the new op count.
"""
import re
from collections import defaultdict

from ..chakra.node import Node


_MB_RE = re.compile(r"^(?:shadow_)?mb(\d+)\.")
_BLOCK_RE = re.compile(r"transformer\.(\d+)\.")
_TENSOR_NAME_RE = re.compile(r"\.([A-Za-z_][A-Za-z_0-9]*)@")


def _is_backward_tensor_name(node_name):
    m = _TENSOR_NAME_RE.search(node_name)
    if m is None:
        return False
    tname = m.group(1)
    return (
        len(tname) >= 2
        and tname[0] == "d"
        and tname[1].isalpha()
        and tname[1].islower()
        and tname != "dff"
    )


def _extract_mb_block(node_name):
    mb = _MB_RE.match(node_name)
    blk = _BLOCK_RE.search(node_name)
    if mb is None or blk is None:
        return None
    return int(mb.group(1)), int(blk.group(1))


class ActivationRecomputePostProcess:
    @classmethod
    def apply(cls, bundled_graph):
        """Inflate backward-pass num_ops in-place, once per unique graph object.

        ``BundledConvertChakra`` aliases every non-zero spatial rank's
        HybridGraph to the corresponding pp-slice zero rank (see
        ``convert_chakra.py``: ``buckets[non_zero] = buckets[zero]``). Iterating
        ``bundled_graph.graphs`` would therefore visit the same object
        ``dp * tp * cp * sp`` times per pp-slice and compound the scaling, so
        the loop dedupes by ``id(hybrid_graph)``.
        """
        print("Applying activation recomputation (inflating backward num_ops)")
        seen_graph_ids = set()
        scaled = 0
        for _readable_rank, hybrid_graph in bundled_graph.graphs.items():
            graph_key = id(hybrid_graph)
            if graph_key in seen_graph_ids:
                continue
            seen_graph_ids.add(graph_key)
            cls._apply_to_rank(hybrid_graph)
            scaled += 1
        print(f"Activation recomputation applied to {scaled} unique graph(s)")
        return bundled_graph

    @classmethod
    def _apply_to_rank(cls, hybrid_graph):
        block_groups = defaultdict(lambda: {"F": [], "B": []})
        for nodes_this_tensor in hybrid_graph.tensor_map_nodes.values():
            for node in nodes_this_tensor.values():
                if node.node_type != Node.NodeType.COMP_NODE:
                    continue
                mb_blk = _extract_mb_block(node.name)
                if mb_blk is None:
                    continue
                is_bwd = _is_backward_tensor_name(node.name)
                block_groups[mb_blk]["B" if is_bwd else "F"].append(node)

        for key, grp in block_groups.items():
            f_nodes = grp["F"]
            b_nodes = grp["B"]
            if not f_nodes or not b_nodes:
                continue
            f_total = sum(getattr(n, "num_ops", 0) or 0 for n in f_nodes)
            b_total = sum(getattr(n, "num_ops", 0) or 0 for n in b_nodes)
            if b_total <= 0 or f_total <= 0:
                continue
            # Standard autodiff: backward GEMMs ≈ 2× forward GEMMs, so
            # f_total / b_total ≈ 0.5 and scale ≈ 1.5. If the ratio is far
            # outside that band the grouping is missing nodes; bail loudly
            # rather than silently inflating FLOPs by orders of magnitude.
            ratio = f_total / b_total
            assert 0.05 < ratio < 2.0, (
                f"activation_recompute: unexpected f/b ratio {ratio:.3f} at "
                f"(mb, block)={key}; expected ~0.5 for a standard Transformer. "
                f"f_total={f_total}, b_total={b_total}"
            )
            scale = 1.0 + ratio
            for bn in b_nodes:
                orig_ops = getattr(bn, "num_ops", 0) or 0
                orig_size = getattr(bn, "tensor_size", 0) or 0
                bn.num_ops = int(orig_ops * scale)
                # Keep OI roughly invariant by scaling tensor_size the same way.
                bn.tensor_size = int(orig_size * scale)
