import copy
import sympy as sp

from symbolic_tensor_graph.graph.connect_graph import ConnectGraph
from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph
from symbolic_tensor_graph.graph.graph import TensorGraph
from symbolic_tensor_graph.graph.grad_updater import FSDPWeightGradManager
from symbolic_tensor_graph.ops import Add, PlaceHolder
from symbolic_tensor_graph.tensor import Tensor
from .llama_model import (
    group_query_attention,
    linear_group_query_attention,
    transformer_decoders,
)
from .utils import reduce_chain


def expert_branch(ffn_path=None, moe_wrapper_path=None):
    if ffn_path is None:
        ffn_path = "./sharding_spreadsheets/module3/tpsp_moe/llama_feed_forward_network.csv"
    if moe_wrapper_path is None:
        moe_wrapper_path = "./sharding_spreadsheets/module3/tpsp_moe/expert_wrapper.csv"

    ffn = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(ffn_path),
        "ffn.%s",
        old_symbol_map_new_symbol={"Seq": "Seq*KExperts/(Experts*ep)"},
    )
    moe_wrapper = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(moe_wrapper_path),
        "ldis.%s",
    )

    expert = ConnectGraph.apply(
        [moe_wrapper, ffn],
        {
            "ldis.x_expert": "ffn.x0",
            "ffn.xdown": "ldis.y_expert",
            "ldis.dy_expert": "ffn.dxdown",
            "ffn.dx0": "ldis.dx_expert",
        },
    )
    return expert


def feed_forward_network(
    symbol_map_value, ffn_path=None, expert_wrapper_path=None, moe_frame_path=None
):
    if moe_frame_path is None:
        moe_frame_path = "./sharding_spreadsheets/module3/tpsp_moe/moe_frame.csv"
    experts, kexperts, ep = sp.symbols("Experts KExperts ep")
    experts = symbol_map_value[experts]
    kexperts = symbol_map_value[kexperts]
    ep = symbol_map_value[ep]
    experts_each_group = experts / ep
    assert experts_each_group == int(experts_each_group)
    experts_each_group = int(experts_each_group)

    expert = expert_branch(ffn_path, expert_wrapper_path)
    moe_frame = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(moe_frame_path), "moe.%s"
    )

    links = dict()
    branches = list()

    for i in range(experts_each_group):
        branches.append(ReplicateGraph.apply(expert, f"moe.{i}.%s"))
        # links["moe.xrouted"] = f"moe.{i}.ldis.x"        # one to multiple, need link multiple times
        # links["moe.dyrouted"] = f"moe.{i}.ldis.dy"
        # links[f"moe.{i}.ldis.dx"] = "moe.dxrouted"      # multiple to one, need reduce nodes
        # links[f"moe.{i}.ldis.y"] = "moe.yrouted"

    moe = ConnectGraph.apply([moe_frame] + branches, links)
    tensor_id_map_tensor = moe.get_tensor_id_map_tensor()

    # one to multiple
    moe_xrouted = tensor_id_map_tensor["moe.xrouted@0"]
    moe_dyrouted = tensor_id_map_tensor["moe.dyrouted@0"]
    for i in range(experts_each_group):
        links = dict()
        links["moe.xrouted"] = f"moe.{i}.ldis.x"
        links["moe.dyrouted"] = f"moe.{i}.ldis.dy"
        moe = ConnectGraph.apply([moe], links, inplace=True)
        moe.out_tensors.append(moe_xrouted)
        moe.out_tensors.append(moe_dyrouted)

    moe.out_tensors.remove(moe_xrouted)
    moe.out_tensors.remove(moe_dyrouted)

    # multiple to one
    to_be_reduce_moe_dxrouted = list()
    to_be_reduce_moe_yrouted = list()

    for i in range(experts_each_group):
        branch_ldis_dx = tensor_id_map_tensor[f"moe.{i}.ldis.dx@0"]
        to_be_reduce_moe_dxrouted.append(branch_ldis_dx)
        moe.out_tensors.remove(branch_ldis_dx)

        branch_ldis_y = tensor_id_map_tensor[f"moe.{i}.ldis.y@0"]
        to_be_reduce_moe_yrouted.append(branch_ldis_y)
        moe.out_tensors.remove(branch_ldis_y)

    # merge those reduce in a chain with 0 ops, which equavilent to a single node
    merged_dxrouted = reduce_chain(to_be_reduce_moe_dxrouted, "moe.dxrouted_r%d", amp=0)
    moe.tensors.extend(merged_dxrouted)
    if len(merged_dxrouted) > 0:
        merged_dxrouted[-1].op_attr = (
            "1"  # last node counts a whole elementwise op, which equalient to a single node
        )
        merged_dxrouted_last = merged_dxrouted[-1]
    else:
        assert len(to_be_reduce_moe_dxrouted) == 1
        merged_dxrouted_last = to_be_reduce_moe_dxrouted[0]
    moe.out_tensors.append(
        merged_dxrouted_last
    )  # add last node as output for future linkage

    merged_yrouted = reduce_chain(to_be_reduce_moe_yrouted, "moe.yrouted_r%d", amp=0)
    moe.tensors.extend(merged_yrouted)
    if len(merged_yrouted) > 0:
        merged_yrouted[-1].op_attr = "1"
        merged_yrouted_last = merged_yrouted[-1]
    else:
        assert len(to_be_reduce_moe_yrouted) == 1
        merged_yrouted_last = to_be_reduce_moe_yrouted[0]
    moe.out_tensors.append(merged_yrouted_last)

    links = {
        merged_dxrouted_last.name: "moe.dxrouted",
        merged_yrouted_last.name: "moe.yrouted",
    }
    moe = ConnectGraph.apply([moe], links)
    return moe


def transformer_decoder_block(
    symbol_map_value, layernorm_path=None, residual_path=None
):
    if layernorm_path is None:
        layernorm_path = "./sharding_spreadsheets/module3/tpsp_moe/layer_norm.csv"
    if residual_path is None:
        residual_path = "./sharding_spreadsheets/module3/tpsp_moe/residual.csv"

    input_layernorm = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(layernorm_path),
        "input_norm.%s",
        old_symbol_map_new_symbol={"tp": "tp"},
    )
    mha = ReplicateGraph.apply(
        group_query_attention(), "mha.%s", old_symbol_map_new_symbol={"tp": "tp"}
    )
    mha_res = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(residual_path),
        "mha_res.%s",
        old_symbol_map_new_symbol={"tp": "tp"},
    )

    post_layernorm = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(layernorm_path),
        "post_attn_norm.%s",
        old_symbol_map_new_symbol={"tp": "tp"},
    )

    ffn = feed_forward_network(symbol_map_value)

    ffn_res = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(residual_path),
        "ffn_res.%s",
        old_symbol_map_new_symbol={"tp": "tp"},
    )

    links = dict()
    # input_layernorm
    links["input_norm.y"] = "mha.x"
    # links["mha_dx"] = "input_norm_dy"

    # mha
    links["mha.o"] = "mha_res.x1"
    links["input_norm.x"] = "mha_res.x2"
    links["mha_res.dx1"] = "mha.do"
    # links["mha_res_dx2"] = "input_norm_dy"

    # mha res
    links["mha_res.y"] = "post_attn_norm.x"
    links["post_attn_norm.dx"] = "mha_res.dy"

    # post_layer_norm
    links["post_attn_norm.y"] = "moe.x"
    # links["moe_dx"] = "post_layer_norm_dy"

    # ffn
    links["moe.y"] = "ffn_res.x1"
    links["post_attn_norm.x"] = "ffn_res.x2"
    links["ffn_res.dx1"] = "moe.dy"
    # links["ffn_res_dx2"] = "post_layer_norm_dy"

    decoder_block = ConnectGraph.apply(
        [input_layernorm, mha, mha_res, post_layernorm, ffn, ffn_res], links
    )

    tensor_id_map_tensor = decoder_block.get_tensor_id_map_tensor()

    input_norm_dy = tensor_id_map_tensor["input_norm.dy@0"]
    assert input_norm_dy.op_type == PlaceHolder.type_name
    input_norm_dy.op_type = Add.type_name
    input_norm_dy.x1 = tensor_id_map_tensor["mha.dx@0"]
    input_norm_dy.x2 = tensor_id_map_tensor["mha_res.dx2@0"]
    input_norm_dy.x2_shape = copy.deepcopy(input_norm_dy.x1_shape)
    input_norm_dy.x2_hidden = copy.deepcopy(input_norm_dy.x1_hidden)
    decoder_block.in_tensors.remove(input_norm_dy)
    decoder_block.out_tensors.remove(input_norm_dy.x1)
    decoder_block.out_tensors.remove(input_norm_dy.x2)

    post_attn_norm_dy = tensor_id_map_tensor["post_attn_norm.dy@0"]
    assert post_attn_norm_dy.op_type == PlaceHolder.type_name
    post_attn_norm_dy.op_type = Add.type_name
    post_attn_norm_dy.x1 = tensor_id_map_tensor["moe.dx@0"]
    post_attn_norm_dy.x2 = tensor_id_map_tensor["ffn_res.dx2@0"]
    post_attn_norm_dy.x2_shape = copy.deepcopy(post_attn_norm_dy.x1_shape)
    post_attn_norm_dy.x2_hidden = copy.deepcopy(post_attn_norm_dy.x1_hidden)
    decoder_block.in_tensors.remove(post_attn_norm_dy)
    decoder_block.out_tensors.remove(post_attn_norm_dy.x1)
    decoder_block.out_tensors.remove(post_attn_norm_dy.x2)

    decoder_block = FSDPWeightGradManager.apply(decoder_block)

    return decoder_block


def transformer(num_layers, symbol_map_value, embedding_path=None, regenerate=False):
    from . import CACHE_DIR
    import os

    experts, kexperts, ep = sp.symbols("Experts KExperts ep")
    experts = symbol_map_value[experts]
    kexperts = symbol_map_value[kexperts]
    ep = symbol_map_value[ep]
    experts_each_group = experts / ep
    cache_filename = os.path.join(
        CACHE_DIR, f"moe_{num_layers}_{experts_each_group}.csv"
    )
    if os.path.exists(cache_filename) and not regenerate:
        return TensorGraph.load_tensor_graph(cache_filename)

    if embedding_path is None:
        embedding_path = "./sharding_spreadsheets/module3/tpsp_moe/embedding.csv"
    in_emb = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(embedding_path),
        "in_emb.%s",
        old_symbol_map_new_symbol={"Din": "Dvocal", "Dout": "Dmodel", "tp": "tp"},
    )
    out_emb = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(embedding_path),
        "out_emb.%s",
        old_symbol_map_new_symbol={"Din": "Dmodel", "Dout": "Dvocal", "tp": "tp"},
    )

    decoder_template = transformer_decoder_block(symbol_map_value)
    decoders = transformer_decoders(num_layers, decoder_template)

    links = dict()
    links["in_emb.y"] = "transformer.0.input_norm.x"
    links["transformer.0.input_norm.dx"] = "in_emb.dy"
    links[f"transformer.{num_layers-1}.ffn_res.y"] = "out_emb.x"
    links["out_emb.dx"] = f"transformer.{num_layers-1}.ffn_res.dy"

    transformer = ConnectGraph.apply([decoders, in_emb, out_emb], links)

    loss = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph("./sharding_spreadsheets/module3/tpsp_moe/loss.csv"),
        "loss.%s",
        old_symbol_map_new_symbol={"tp": "tp"},
    )
    links = dict()
    links["out_emb.y"] = "loss.y"
    links["loss.dy"] = "out_emb.dy"
    transformer = ConnectGraph.apply([transformer, loss], links)

    transformer.save_tensor_graph(cache_filename)
    return transformer


# ---------------------------------------------------------------------------
#  Change 4: Optimized batched expert construction for large expert counts
# ---------------------------------------------------------------------------

_BATCHED_EXPERT_THRESHOLD = 32


def feed_forward_network_auto(
    symbol_map_value, ffn_path=None, expert_wrapper_path=None, moe_frame_path=None
):
    """Choose per-expert loop or batched mode based on experts_per_group."""
    experts_sym, ep_sym = sp.symbols("Experts ep")
    experts_val = symbol_map_value[experts_sym]
    ep_val = symbol_map_value[ep_sym]
    experts_each_group = int(experts_val / ep_val)

    if experts_each_group <= _BATCHED_EXPERT_THRESHOLD:
        return feed_forward_network(
            symbol_map_value, ffn_path, expert_wrapper_path, moe_frame_path
        )
    return feed_forward_network_batched(
        symbol_map_value, ffn_path, moe_frame_path
    )


def feed_forward_network_batched(
    symbol_map_value, ffn_path=None, moe_frame_path=None
):
    """Represent all local experts as a single batched FFN (approximate).

    Instead of creating Experts/ep individual branches (which generates
    thousands of graph nodes), this creates ONE FFN whose token count
    covers all local experts, yielding correct total FLOPs.  Weight
    memory is compensated via an extra placeholder node.
    """
    if ffn_path is None:
        ffn_path = "./sharding_spreadsheets/module3/tpsp_moe/llama_feed_forward_network.csv"
    if moe_frame_path is None:
        moe_frame_path = "./sharding_spreadsheets/module3/tpsp_moe/moe_frame.csv"

    experts_sym, kexperts_sym, ep_sym = sp.symbols("Experts KExperts ep")
    experts_val = symbol_map_value[experts_sym]
    ep_val = symbol_map_value[ep_sym]
    experts_each_group = int(experts_val / ep_val)

    moe_frame = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(moe_frame_path), "moe.%s"
    )

    batched_ffn = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(ffn_path),
        "moe.batched_ffn.%s",
        old_symbol_map_new_symbol={"Seq": "Seq*KExperts/ep"},
    )

    links = {
        "moe.xrouted": "moe.batched_ffn.x0",
        "moe.batched_ffn.xdown": "moe.yrouted",
        "moe.dyrouted": "moe.batched_ffn.dxdown",
        "moe.batched_ffn.dx0": "moe.dxrouted",
    }
    moe = ConnectGraph.apply([moe_frame, batched_ffn], links)

    if experts_each_group > 1:
        weight_mem_node = Tensor(create_empty=True)
        weight_mem_node.name = "moe.extra_expert_weights"
        weight_mem_node.revision = 0
        weight_mem_node.require_grads = True
        weight_mem_node.op_type = "T"
        weight_mem_node.x1_shape = [
            sp.parse_expr(f"({experts_each_group}-1)*Dmodel"),
            sp.parse_expr("Dff"),
        ]
        weight_mem_node.x1_hidden = [sp.Integer(1)]
        weight_mem_node.extra_attr = {}
        moe.tensors.append(weight_mem_node)
        moe.in_tensors.append(weight_mem_node)

    return moe


# ---------------------------------------------------------------------------
#  Change 2 & 3: Qwen3-style decoder block with heterogeneous attention
#                and optional shared expert
# ---------------------------------------------------------------------------

def transformer_decoder_block_qwen3(
    symbol_map_value,
    attention_type="full",
    shared_expert_dff=0,
    attention_backend=None,
    layernorm_path=None,
    residual_path=None,
):
    """Decoder block supporting linear / full attention and shared expert."""
    if layernorm_path is None:
        layernorm_path = "./sharding_spreadsheets/module3/tpsp_moe/layer_norm.csv"
    if residual_path is None:
        residual_path = "./sharding_spreadsheets/module3/tpsp_moe/residual.csv"

    input_layernorm = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(layernorm_path),
        "input_norm.%s",
        old_symbol_map_new_symbol={"tp": "tp"},
    )

    if attention_type == "linear":
        mha = ReplicateGraph.apply(
            linear_group_query_attention(),
            "mha.%s",
            old_symbol_map_new_symbol={"tp": "tp"},
        )
    else:
        mha = ReplicateGraph.apply(
            group_query_attention(attention_backend=attention_backend),
            "mha.%s",
            old_symbol_map_new_symbol={"tp": "tp"},
        )

    mha_res = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(residual_path),
        "mha_res.%s",
        old_symbol_map_new_symbol={"tp": "tp"},
    )
    post_layernorm = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(layernorm_path),
        "post_attn_norm.%s",
        old_symbol_map_new_symbol={"tp": "tp"},
    )

    ffn = feed_forward_network_auto(symbol_map_value)

    ffn_res = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(residual_path),
        "ffn_res.%s",
        old_symbol_map_new_symbol={"tp": "tp"},
    )

    links = dict()
    links["input_norm.y"] = "mha.x"
    links["mha.o"] = "mha_res.x1"
    links["input_norm.x"] = "mha_res.x2"
    links["mha_res.dx1"] = "mha.do"
    links["mha_res.y"] = "post_attn_norm.x"
    links["post_attn_norm.dx"] = "mha_res.dy"
    links["post_attn_norm.y"] = "moe.x"
    links["post_attn_norm.x"] = "ffn_res.x2"

    components = [input_layernorm, mha, mha_res, post_layernorm, ffn, ffn_res]

    if shared_expert_dff > 0:
        shared_ffn_path = (
            "./sharding_spreadsheets/module3/tpsp_moe/llama_feed_forward_network.csv"
        )
        shared_ffn = ReplicateGraph.apply(
            TensorGraph.load_tensor_graph(shared_ffn_path),
            "shared_ffn.%s",
            old_symbol_map_new_symbol={"Dff": "SharedDff"},
        )
        components.append(shared_ffn)
    else:
        links["moe.y"] = "ffn_res.x1"
        links["ffn_res.dx1"] = "moe.dy"

    decoder_block = ConnectGraph.apply(components, links)
    tensor_id_map = decoder_block.get_tensor_id_map_tensor()

    if shared_expert_dff > 0:
        # --- forward one-to-many: post_attn_norm.y → {moe.x, shared_ffn.x0}
        post_y = tensor_id_map["post_attn_norm.y@0"]
        decoder_block.out_tensors.append(post_y)
        decoder_block = ConnectGraph.apply(
            [decoder_block], {"post_attn_norm.y": "shared_ffn.x0"}, inplace=True
        )
        tensor_id_map = decoder_block.get_tensor_id_map_tensor()

        # --- forward: ffn_res.x1 = moe.y + shared_ffn.xdown
        ffn_res_x1 = tensor_id_map["ffn_res.x1@0"]
        moe_y = tensor_id_map["moe.y@0"]
        shared_xdown = tensor_id_map["shared_ffn.xdown@0"]

        orig_x1_shape = copy.deepcopy(ffn_res_x1.x1_shape)
        orig_x1_hidden = copy.deepcopy(ffn_res_x1.x1_hidden)

        ffn_res_x1.op_type = Add.type_name
        ffn_res_x1.x1 = moe_y
        ffn_res_x1.x2 = shared_xdown
        ffn_res_x1.x1_shape = orig_x1_shape
        ffn_res_x1.x1_hidden = orig_x1_hidden
        ffn_res_x1.x2_shape = copy.deepcopy(orig_x1_shape)
        ffn_res_x1.x2_hidden = copy.deepcopy(orig_x1_hidden)
        decoder_block.in_tensors.remove(ffn_res_x1)
        decoder_block.out_tensors.remove(moe_y)
        decoder_block.out_tensors.remove(shared_xdown)

        # --- backward one-to-many: ffn_res.dx1 → {moe.dy, shared_ffn.dxdown}
        ffn_res_dx1 = tensor_id_map["ffn_res.dx1@0"]
        decoder_block = ConnectGraph.apply(
            [decoder_block], {"ffn_res.dx1": "moe.dy"}, inplace=True
        )
        decoder_block.out_tensors.append(ffn_res_dx1)
        decoder_block = ConnectGraph.apply(
            [decoder_block], {"ffn_res.dx1": "shared_ffn.dxdown"}, inplace=True
        )
        tensor_id_map = decoder_block.get_tensor_id_map_tensor()

    # --- gradient merging: input_norm.dy = mha.dx + mha_res.dx2
    input_norm_dy = tensor_id_map["input_norm.dy@0"]
    assert input_norm_dy.op_type == PlaceHolder.type_name
    input_norm_dy.op_type = Add.type_name
    input_norm_dy.x1 = tensor_id_map["mha.dx@0"]
    input_norm_dy.x2 = tensor_id_map["mha_res.dx2@0"]
    input_norm_dy.x2_shape = copy.deepcopy(input_norm_dy.x1_shape)
    input_norm_dy.x2_hidden = copy.deepcopy(input_norm_dy.x1_hidden)
    decoder_block.in_tensors.remove(input_norm_dy)
    decoder_block.out_tensors.remove(input_norm_dy.x1)
    decoder_block.out_tensors.remove(input_norm_dy.x2)

    # --- gradient merging: post_attn_norm.dy
    post_attn_norm_dy = tensor_id_map["post_attn_norm.dy@0"]
    assert post_attn_norm_dy.op_type == PlaceHolder.type_name

    if shared_expert_dff > 0:
        # Three gradient sources: moe.dx, ffn_res.dx2, shared_ffn.dx0
        # Use the placeholder's original shape for all Add nodes (the
        # communication matcher resolves mismatches later).
        orig_shape = copy.deepcopy(post_attn_norm_dy.x1_shape)
        orig_hidden = copy.deepcopy(post_attn_norm_dy.x1_hidden)

        moe_dx = tensor_id_map["moe.dx@0"]
        ffn_res_dx2 = tensor_id_map["ffn_res.dx2@0"]
        shared_dx0 = tensor_id_map["shared_ffn.dx0@0"]

        temp_add = Tensor(create_empty=True)
        temp_add.name = "post_attn_norm._grad_merge"
        temp_add.revision = 0
        temp_add.require_grads = False
        temp_add.op_type = Add.type_name
        temp_add.x1 = moe_dx
        temp_add.x2 = ffn_res_dx2
        temp_add.x1_shape = copy.deepcopy(orig_shape)
        temp_add.x1_hidden = copy.deepcopy(orig_hidden)
        temp_add.x2_shape = copy.deepcopy(orig_shape)
        temp_add.x2_hidden = copy.deepcopy(orig_hidden)
        temp_add.extra_attr = {}
        decoder_block.tensors.append(temp_add)

        post_attn_norm_dy.op_type = Add.type_name
        post_attn_norm_dy.x1 = temp_add
        post_attn_norm_dy.x2 = shared_dx0
        post_attn_norm_dy.x1_shape = copy.deepcopy(orig_shape)
        post_attn_norm_dy.x1_hidden = copy.deepcopy(orig_hidden)
        post_attn_norm_dy.x2_shape = copy.deepcopy(orig_shape)
        post_attn_norm_dy.x2_hidden = copy.deepcopy(orig_hidden)
        decoder_block.in_tensors.remove(post_attn_norm_dy)
        decoder_block.out_tensors.remove(moe_dx)
        decoder_block.out_tensors.remove(ffn_res_dx2)
        decoder_block.out_tensors.remove(shared_dx0)
    else:
        post_attn_norm_dy.op_type = Add.type_name
        post_attn_norm_dy.x1 = tensor_id_map["moe.dx@0"]
        post_attn_norm_dy.x2 = tensor_id_map["ffn_res.dx2@0"]
        post_attn_norm_dy.x2_shape = copy.deepcopy(post_attn_norm_dy.x1_shape)
        post_attn_norm_dy.x2_hidden = copy.deepcopy(post_attn_norm_dy.x1_hidden)
        decoder_block.in_tensors.remove(post_attn_norm_dy)
        decoder_block.out_tensors.remove(post_attn_norm_dy.x1)
        decoder_block.out_tensors.remove(post_attn_norm_dy.x2)

    decoder_block = FSDPWeightGradManager.apply(decoder_block)
    return decoder_block


# ---------------------------------------------------------------------------
#  Change 2: Heterogeneous layer stacking (mixed linear + full attention)
# ---------------------------------------------------------------------------

def transformer_decoders_heterogeneous(num_layers, layer_types, templates):
    """Stack decoder layers using per-layer attention type selection.

    Args:
        num_layers: total number of decoder layers.
        layer_types: list of length num_layers, each "full" or "linear".
        templates: dict mapping attention type name to a decoder block template.
    """
    links = dict()
    decoders = list()
    for i in range(num_layers):
        template = templates[layer_types[i]]
        decoder = ReplicateGraph.apply(template, f"transformer.{i}.%s")
        decoders.append(decoder)
        if i > 0:
            links[f"transformer.{i-1}.ffn_res.y"] = f"transformer.{i}.input_norm.x"
            links[f"transformer.{i}.input_norm.dx"] = f"transformer.{i-1}.ffn_res.dy"

    decoders = ConnectGraph.apply(decoders, links)
    return decoders


# ---------------------------------------------------------------------------
#  Full Qwen3-MoE transformer builder
# ---------------------------------------------------------------------------

def transformer_qwen3(
    num_layers,
    symbol_map_value,
    layer_types=None,
    shared_expert_dff=0,
    attention_backend=None,
    embedding_path=None,
    regenerate=False,
):
    from . import CACHE_DIR
    import os

    experts_sym, ep_sym = sp.symbols("Experts ep")
    experts_val = symbol_map_value[experts_sym]
    ep_val = symbol_map_value[ep_sym]
    experts_each_group = int(experts_val / ep_val)

    layer_types_tag = "hetero" if layer_types else "homo"
    shared_tag = f"se{shared_expert_dff}" if shared_expert_dff else "nose"
    cache_filename = os.path.join(
        CACHE_DIR,
        f"qwen3moe_{num_layers}_{experts_each_group}_{layer_types_tag}_{shared_tag}.csv",
    )
    if os.path.exists(cache_filename) and not regenerate:
        return TensorGraph.load_tensor_graph(cache_filename)

    if embedding_path is None:
        embedding_path = "./sharding_spreadsheets/module3/tpsp_moe/embedding.csv"
    in_emb = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(embedding_path),
        "in_emb.%s",
        old_symbol_map_new_symbol={"Din": "Dvocal", "Dout": "Dmodel", "tp": "tp"},
    )
    out_emb = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(embedding_path),
        "out_emb.%s",
        old_symbol_map_new_symbol={"Din": "Dmodel", "Dout": "Dvocal", "tp": "tp"},
    )

    if layer_types is None:
        layer_types = ["full"] * num_layers

    if len(layer_types) < num_layers:
        reps = (num_layers + len(layer_types) - 1) // len(layer_types)
        layer_types = (layer_types * reps)[:num_layers]

    unique_types = set(layer_types)
    templates = {}
    for atype in unique_types:
        print(f"  Building {atype}-attention decoder template ...")
        templates[atype] = transformer_decoder_block_qwen3(
            symbol_map_value,
            attention_type=atype,
            shared_expert_dff=shared_expert_dff,
            attention_backend=attention_backend,
        )

    print("Stacking heterogeneous decoder layers ...")
    decoders = transformer_decoders_heterogeneous(num_layers, layer_types, templates)

    links = dict()
    links["in_emb.y"] = "transformer.0.input_norm.x"
    links["transformer.0.input_norm.dx"] = "in_emb.dy"
    links[f"transformer.{num_layers-1}.ffn_res.y"] = "out_emb.x"
    links["out_emb.dx"] = f"transformer.{num_layers-1}.ffn_res.dy"

    model = ConnectGraph.apply([decoders, in_emb, out_emb], links)

    loss = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(
            "./sharding_spreadsheets/module3/tpsp_moe/loss.csv"
        ),
        "loss.%s",
        old_symbol_map_new_symbol={"tp": "tp"},
    )
    links = dict()
    links["out_emb.y"] = "loss.y"
    links["loss.dy"] = "out_emb.dy"
    model = ConnectGraph.apply([model, loss], links)

    model.save_tensor_graph(cache_filename)
    return model
