import os
import argparse
import sympy as sp
from symbolic_tensor_graph.graph.graph import TensorGraph
from symbolic_tensor_graph.graph.grad_updater import (
    GradUpdater,
    LocalSGDIterationPostProcess,
    MicroBatchReplicator,
    MicroBatchReplicatorPostProcess,
)
from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph
from symbolic_tensor_graph.graph.graph_distributer import GraphDistributer
from symbolic_tensor_graph.graph.convert_chakra import BundledConvertChakra
import re
from symbolic_tensor_graph.vram_counting import _print_gpu_vram

mixprecision = False


def str_to_bool(v):
    # Convert "true" to True and "false" to False
    return v.lower() in ("true", "t", "1", "yes", "y")


def _create_pipeline_tensor_map_mix_precision(
    _tensors, _temporal_parallel_dims, _symbol_map_value, num_stacks
):
    _tensor_map = dict()
    assert len(_temporal_parallel_dims) == 1
    parallel_dim = _temporal_parallel_dims[0]
    range_ = _symbol_map_value[parallel_dim]

    # Determine how many transformer blocks belong to each pipeline stage
    num_stacks_each_stage = [num_stacks // range_] * range_
    for i in range(num_stacks % range_):
        num_stacks_each_stage[i] += 1  # distribute remainder to early stages
    # Cumulative upper bounds for easy stage lookup
    cumulative = []
    acc = 0
    for v in num_stacks_each_stage:
        acc += v
        cumulative.append(acc)

    for tensor in _tensors:
        tid = tensor.id
        # ------------------------------------------------------------------
        # 1) Transformer block tensors
        # ------------------------------------------------------------------
        m = re.search(r"transformer\.(\d+)", tid)
        if m:
            block_idx = int(m.group(1))
            # Find the first cumulative upper bound that exceeds block_idx
            stage = next(i for i, up in enumerate(cumulative) if block_idx < up)
            _tensor_map[tid] = {parallel_dim: stage}
            continue

        # ------------------------------------------------------------------
        # 2) Special tensors (embeddings, loss etc.)
        # ------------------------------------------------------------------
        if "in_emb" in tid:
            _tensor_map[tid] = {parallel_dim: 0}
        elif "out_emb" in tid or "loss" in tid:
            _tensor_map[tid] = {parallel_dim: (range_ - 1)}
        else:
            # Any tensor that doesn't match the above categories should be
            # impossible – raise explicit error to catch new patterns early.
            raise ValueError(f"Unrecognized tensor id for pipeline mapping: {tid}")

    return _tensor_map


def _create_pipeline_tensor_map(
    _tensors, _temporal_parallel_dims, _symbol_map_value, num_stacks
):
    if mixprecision:
        return _create_pipeline_tensor_map_mix_precision(
            _tensors, _temporal_parallel_dims, _symbol_map_value, num_stacks
        )
    _tensor_map = dict()
    assert len(_temporal_parallel_dims) == 1
    parallel_dim = _temporal_parallel_dims[0]
    range_ = _symbol_map_value[parallel_dim]
    num_stacks_each_stage = list()
    for i in range(range_):
        num_stacks_each_stage.append(num_stacks // range_)
    for i in range(num_stacks - range_ * (num_stacks // range_)):
        num_stacks_each_stage[i] += 1
    for i in range(range_):
        if i == 0:
            continue
        num_stacks_each_stage[i] += num_stacks_each_stage[i - 1]
    # num_stacks_each_stage.append(num_stacks_each_stage[-1]+100000)

    for tensor in _tensors:
        if tensor.id == "transformer.18._sharded_weight@1":
            pass
        found = False
        for num_stack in range(num_stacks):
            if f"transformer.{num_stack}." in tensor.id:
                for stage, upper_bound in enumerate(num_stacks_each_stage):
                    if num_stack < upper_bound:
                        _tensor_map[tensor.id] = {parallel_dim: stage}
                        found = True
                        break
                if found:
                    break
        if found:
            pass
        elif "in_emb" in tensor.id:
            _tensor_map[tensor.id] = {parallel_dim: 0}
        elif "out_emb" in tensor.id:
            _tensor_map[tensor.id] = {parallel_dim: (num_stacks - 1) % range_}
        elif "loss" in tensor.id:
            _tensor_map[tensor.id] = {parallel_dim: (num_stacks - 1) % range_}
        else:
            assert False, tensor.name
    return _tensor_map


def _postprocess_chakra_graph(chakra_graph, args, dp):
    if os.environ.get("STAGE_MICROBATCH_OPTIMIZE", "0") != "0":
        chakra_graph = MicroBatchReplicatorPostProcess.apply(
            chakra_graph, args.batch // args.micro_batch
        )
    if args.num_iterations > 1 or args.dp_local_sgd_interval > 1:
        chakra_graph = LocalSGDIterationPostProcess.apply(
            chakra_graph,
            num_iterations=args.num_iterations,
            sync_interval=args.dp_local_sgd_interval,
            dp_symbol=dp,
        )
    return chakra_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, help="dir where stores output traces", required=True
    )
    parser.add_argument(
        "--output_name", type=str, help="name of output traces", required=True
    )
    parser.add_argument(
        "--dp", type=int, help="data parallel degree", required=False, default=1
    )
    parser.add_argument(
        "--tp", type=int, help="tensor parallel degree", required=False, default=1
    )
    parser.add_argument(
        "--sp", type=int, help="sequence parallel degree", required=False, default=1
    )
    parser.add_argument(
        "--ep", type=int, help="expert parallel degree", required=False, default=1
    )
    parser.add_argument(
        "--pp", type=int, default=1, help="pipeline parallel degree", required=False
    )
    parser.add_argument(
        "--weight_sharded",
        type=str_to_bool,
        help="whether weight sharded",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--activation_recompute",
        type=str_to_bool,
        help="whether recompute activation",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--tpsp",
        type=str_to_bool,
        help="use tp+sp or tp only",
        required=False,
        default=True,
    )
    parser.add_argument("--dvocal", type=int, default=32000, required=False)
    parser.add_argument("--dmodel", type=int, default=8192, required=False)
    parser.add_argument("--dff", type=int, default=28672, required=False)
    parser.add_argument("--batch", type=int, default=64, required=False)
    parser.add_argument("--micro_batch", type=int, default=-1, required=False)
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=1,
        required=False,
        help="number of training iterations to emit in the workload",
    )
    parser.add_argument(
        "--dp_local_sgd_interval",
        type=int,
        default=1,
        required=False,
        help="perform DP all-reduce every K iterations; 1 keeps synchronous DP",
    )
    parser.add_argument("--seq", type=int, default=1024, required=False)
    parser.add_argument("--head", type=int, default=64, required=False)
    parser.add_argument("--kvhead", type=int, default=8, required=False)
    parser.add_argument("--num_stacks", type=int, default=80, required=False)
    parser.add_argument("--experts", type=int, default=8, required=False)
    parser.add_argument("--kexperts", type=int, default=2, required=False)
    parser.add_argument(
        "--chakra_schema_version", type=str, default="v0.0.4", required=False
    )
    parser.add_argument("--model_type", type=str, default="dense", required=False)
    parser.add_argument(
        "--mixed_precision", type=str_to_bool, default=False, required=False
    )
    parser.add_argument(
        "--print_gpu_vram",
        type=str_to_bool,
        default=False,
        required=False,
        help="Whether to print per-GPU VRAM footprint (total / params / acts / grads) in GiB",
    )
    parser.add_argument(
        "--flash_attention",
        type=str_to_bool,
        default=False,
        required=False,
        help="Use FlashAttention kernel with O(S^2) FLOPs and no S×S materialization",
    )
    parser.add_argument(
        "--attention_backend",
        type=str,
        default="auto",
        choices=["auto", "standard", "fused", "flash"],
        required=False,
        help="Attention kernel backend. 'auto' preserves the legacy --flash_attention behavior.",
    )

    # Qwen3-MoE specific parameters
    parser.add_argument(
        "--linear_num_key_heads",
        type=int,
        default=0,
        required=False,
        help="Number of key/query heads for linear attention layers (0 = disabled)",
    )
    parser.add_argument(
        "--linear_num_value_heads",
        type=int,
        default=0,
        required=False,
        help="Number of value heads for linear attention layers",
    )
    parser.add_argument(
        "--linear_head_dim",
        type=int,
        default=128,
        required=False,
        help="Head dimension for linear attention layers",
    )
    parser.add_argument(
        "--conv_kernel_dim",
        type=int,
        default=4,
        required=False,
        help="Causal conv1d kernel size for linear attention",
    )
    parser.add_argument(
        "--shared_expert_dff",
        type=int,
        default=0,
        required=False,
        help="Shared expert FFN intermediate dim (0 = no shared expert)",
    )
    parser.add_argument(
        "--layer_types",
        type=str,
        default="",
        required=False,
        help='Comma-separated attention types per layer, e.g. "linear,linear,linear,full". '
        "Cycled to match num_stacks.",
    )

    args = parser.parse_args()
    if args.num_iterations < 1:
        raise ValueError("--num_iterations must be at least 1")
    if args.dp_local_sgd_interval < 1:
        raise ValueError("--dp_local_sgd_interval must be at least 1")
    if args.flash_attention and args.attention_backend not in ("auto", "flash"):
        raise ValueError(
            "--flash_attention true cannot be combined with --attention_backend "
            f"{args.attention_backend}"
        )

    attention_backend = args.attention_backend
    if attention_backend == "auto":
        attention_backend = "flash" if args.flash_attention else "fused"

    os.makedirs(args.output_dir, exist_ok=True)
    if not "%d" in args.output_name:
        args.output_name = f"{args.output_name}.%d.et"
    generated_filename = os.path.join(args.output_dir, args.output_name)
    dp, tp, pp, spp, ep, fsdp = sp.symbols("dp tp pp cp ep fsdp")
    (
        Din,
        Dout,
        Dmodel,
        Dff,
        Batch,
        Seq,
        Head,
        KVHead,
        Experts,
        KExperts,
        Dvocal,
        MicroBatch,
    ) = sp.symbols(
        "Din Dout Dmodel Dff Batch Seq Head KVHead Experts KExperts Dvocal MicroBatch"
    )
    LKHead, LVHead, LHeadDim, ConvKernel, SharedDff = sp.symbols(
        "LKHead LVHead LHeadDim ConvKernel SharedDff"
    )
    if args.micro_batch == -1:
        args.micro_batch = args.batch
    symbol_map_value = {
        Dvocal: args.dvocal,
        Dmodel: args.dmodel,
        Dff: args.dff,
        Batch: args.batch,
        MicroBatch: args.micro_batch,
        Seq: args.seq,
        Head: args.head,
        KVHead: args.kvhead,
        Experts: args.experts,
        KExperts: args.kexperts,
        dp: args.dp,
        tp: args.tp,
        pp: args.pp,
        spp: args.sp,
        ep: args.ep,
    }
    if args.linear_num_key_heads > 0:
        symbol_map_value[LKHead] = args.linear_num_key_heads
        symbol_map_value[LVHead] = args.linear_num_value_heads
        symbol_map_value[LHeadDim] = args.linear_head_dim
        symbol_map_value[ConvKernel] = args.conv_kernel_dim
    if args.shared_expert_dff > 0:
        symbol_map_value[SharedDff] = args.shared_expert_dff
    num_stacks = args.num_stacks
    temporal_parallel_dims = [pp]
    if args.weight_sharded:
        symbol_map_value[fsdp] = args.dp if args.dp != 0 else 1
        symbol_map_value["fsdp"] = args.dp if args.dp != 0 else 1
    else:
        symbol_map_value[fsdp] = 1
        symbol_map_value["fsdp"] = 1

    hook = 1
    global mixprecision
    if args.mixed_precision:
        mixprecision = True

    if args.model_type == "llama" or args.model_type == "dense":
        if mixprecision:
            from models.stage1.llama_model import llama as transformer_dense
        else:
            from models.stage1.gpt_model import gpt as transformer_dense

        print("Assembling dense model")
        transformer_dense = transformer_dense(
            num_stacks, regenerate=True, tpsp=args.tpsp,
            flash_attention=args.flash_attention,
            attention_backend=attention_backend,
        )
        if os.environ.get("STAGE_MICROBATCH_OPTIMIZE", "0") == "0":
            transformer_dense = MicroBatchReplicator.apply(
                transformer_dense, symbol_map_value
            )
        else:
            print("[Warning] MICROBATCH OPTIMIZE sometimes generate incorrect graphs, use with caution!")
            transformer_dense = ReplicateGraph.apply(
                transformer_dense,
                inplace=True,
                old_symbol_map_new_symbol={"Batch": "MicroBatch"},
            )

        if args.weight_sharded:
            transformer_dense = ReplicateGraph.apply(
                transformer_dense,
                inplace=True,
                old_symbol_map_new_symbol={"fsdp": "dp"},
            )
        else:
            transformer_dense = ReplicateGraph.apply(
                transformer_dense, inplace=True, old_symbol_map_new_symbol={"fsdp": 1}
            )

        # transformer_dense.visualize("dense")
        # transformer_dense.save_tensor_graph("llama.csv")

        transformer_dense = GradUpdater.apply(transformer_dense, inplace=True)
        spatial_parallel_dims_dense = [dp, tp, spp]

        symbol_map_value[tp] *= symbol_map_value[ep]
        # dense model
        pipeline_tensor_map = _create_pipeline_tensor_map(
            transformer_dense.tensors,
            temporal_parallel_dims,
            symbol_map_value,
            num_stacks,
        )

        print("Dense model: Distributing")
        distributed_tensor_graph_dense = GraphDistributer.apply(
            transformer_dense,
            symbol_map_value,
            spatial_parallel_dims_dense,
            temporal_parallel_dims,
            pipeline_tensor_map,
        )

        if args.print_gpu_vram:
            _print_gpu_vram(
                distributed_tensor_graph_dense,
                symbol_map_value,
                mixed_precision=args.mixed_precision,
                header="[Dense] ",
            )

        print("Dense model: Converting Chakra")
        comm_group_file = args.output_name.replace(".%d", "").replace(".et", ".json")
        distributed_chakra_graph_dense = BundledConvertChakra.apply(
            distributed_tensor_graph_dense,
            symbol_map_value,
            os.path.join(args.output_dir, comm_group_file),
            mixed_precision=args.mixed_precision,
        )

        from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend import (
            Chakra004Backend as ReadoutBackend,
        )

        distributed_chakra_graph_dense = _postprocess_chakra_graph(
            distributed_chakra_graph_dense, args, dp
        )

        print("Dense model: reading out")
        distributed_chakra_graph_dense.readout(
            generated_filename, backend=ReadoutBackend
        )
    elif args.model_type == "gpt":
        from models.stage1.gpt_model import gpt as transformer_dense

        print("Assembling dense model")
        transformer_dense = transformer_dense(
            num_stacks, regenerate=True, tpsp=args.tpsp,
            flash_attention=args.flash_attention,
            attention_backend=attention_backend,
        )
        if os.environ.get("STAGE_MICROBATCH_OPTIMIZE", "0") == "0":
            transformer_dense = MicroBatchReplicator.apply(
                transformer_dense, symbol_map_value
            )
        else:
            print("[Warning] MICROBATCH OPTIMIZE sometimes generate incorrect graphs, use with caution!")
            transformer_dense = ReplicateGraph.apply(
                transformer_dense,
                inplace=True,
                old_symbol_map_new_symbol={"Batch": "MicroBatch"},
            )

        if args.weight_sharded:
            transformer_dense = ReplicateGraph.apply(
                transformer_dense,
                inplace=True,
                old_symbol_map_new_symbol={"fsdp": "dp"},
            )
        else:
            transformer_dense = ReplicateGraph.apply(
                transformer_dense, inplace=True, old_symbol_map_new_symbol={"fsdp": 1}
            )

        # transformer_dense.visualize("dense")
        # transformer_dense.save_tensor_graph("gpt.csv")

        transformer_dense = GradUpdater.apply(transformer_dense, inplace=True)
        spatial_parallel_dims_dense = [dp, tp, spp]

        symbol_map_value[tp] *= symbol_map_value[ep]
        # dense model
        pipeline_tensor_map = _create_pipeline_tensor_map(
            transformer_dense.tensors,
            temporal_parallel_dims,
            symbol_map_value,
            num_stacks,
        )

        print("Dense model: Distributing")
        distributed_tensor_graph_dense = GraphDistributer.apply(
            transformer_dense,
            symbol_map_value,
            spatial_parallel_dims_dense,
            temporal_parallel_dims,
            pipeline_tensor_map,
        )

        if args.print_gpu_vram:
            _print_gpu_vram(
                distributed_tensor_graph_dense,
                symbol_map_value,
                mixed_precision=args.mixed_precision,
                header="[GPT] ",
            )

        print("Dense model: Converting Chakra")
        comm_group_file = args.output_name.replace(".%d", "").replace(".et", ".json")
        distributed_chakra_graph_dense = BundledConvertChakra.apply(
            distributed_tensor_graph_dense,
            symbol_map_value,
            os.path.join(args.output_dir, comm_group_file),
            mixed_precision=args.mixed_precision,
        )

        from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend import (
            Chakra004Backend as ReadoutBackend,
        )

        print("Dense model: reading out")
        distributed_chakra_graph_dense = _postprocess_chakra_graph(
            distributed_chakra_graph_dense, args, dp
        )
        distributed_chakra_graph_dense.readout(
            generated_filename, backend=ReadoutBackend
        )

    elif args.model_type == "moe":
        from models.stage1.moe_model import transformer as transformer_moe

        assert args.tpsp
        print("Assembling moe model")
        transformer_moe = transformer_moe(num_stacks, symbol_map_value, regenerate=True)
        if os.environ.get("STAGE_MICROBATCH_OPTIMIZE", "0") == "0":
            transformer_moe = MicroBatchReplicator.apply(
                transformer_moe, symbol_map_value
            )
        else:
            print("[Warning] MICROBATCH OPTIMIZE sometimes generate incorrect graphs, use with caution!")
            assert False, "disable for now"
        transformer_moe = ReplicateGraph.apply(
            transformer_moe,
            inplace=True,
            old_symbol_map_new_symbol={"Batch": "MicroBatch"},
        )

        if args.weight_sharded:
            transformer_moe = ReplicateGraph.apply(
                transformer_moe,
                inplace=True,
                old_symbol_map_new_symbol={"fsdp": "dp"},
            )
        else:
            transformer_moe = ReplicateGraph.apply(
                transformer_moe, inplace=True, old_symbol_map_new_symbol={"fsdp": 1}
            )

        # transformer_moe.visualize("moe")
        # transformer_moe.save_tensor_graph("moe.csv")
        transformer_moe = GradUpdater.apply(transformer_moe, inplace=True)
        spatial_parallel_dims_moe = [dp, tp, spp, ep]

        # moe model
        pipeline_tensor_map = _create_pipeline_tensor_map(
            transformer_moe.tensors,
            temporal_parallel_dims,
            symbol_map_value,
            num_stacks,
        )

        print("MoE model: Distributing")
        distributed_tensor_graph_moe = GraphDistributer.apply(
            transformer_moe,
            symbol_map_value,
            spatial_parallel_dims_moe,
            temporal_parallel_dims,
            pipeline_tensor_map,
        )

        if args.print_gpu_vram:
            _print_gpu_vram(
                distributed_tensor_graph_moe,
                symbol_map_value,
                mixed_precision=args.mixed_precision,
                header="[MoE] ",
            )

        print("MoE model: Converting Chakra")
        comm_group_file = args.output_name.replace(".%d", "").replace(".et", ".json")
        distributed_chakra_graph_moe = BundledConvertChakra.apply(
            distributed_tensor_graph_moe,
            symbol_map_value,
            os.path.join(args.output_dir, comm_group_file),
            mixed_precision=args.mixed_precision,
        )

        from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend import (
            Chakra004Backend as ReadoutBackend,
        )

        print("MoE model: reading out")
        distributed_chakra_graph_moe = _postprocess_chakra_graph(
            distributed_chakra_graph_moe, args, dp
        )
        distributed_chakra_graph_moe.readout(generated_filename, backend=ReadoutBackend)

    elif args.model_type == "qwen3_moe":
        from models.stage1.moe_model import transformer_qwen3 as qwen3_builder

        layer_types_list = None
        if args.layer_types:
            layer_types_list = [t.strip() for t in args.layer_types.split(",")]

        assert args.tpsp
        print("Assembling Qwen3-MoE model")
        transformer_qwen3 = qwen3_builder(
            num_stacks,
            symbol_map_value,
            layer_types=layer_types_list,
            shared_expert_dff=args.shared_expert_dff,
            attention_backend=attention_backend,
            regenerate=True,
        )
        if os.environ.get("STAGE_MICROBATCH_OPTIMIZE", "0") == "0":
            transformer_qwen3 = MicroBatchReplicator.apply(
                transformer_qwen3, symbol_map_value
            )
        else:
            print("[Warning] MICROBATCH OPTIMIZE sometimes generate incorrect graphs, use with caution!")
            assert False, "disable for now"
        # MicroBatchReplicator already substitutes Batch→MicroBatch for
        # activation tensors.  A second ReplicateGraph pass with the same
        # substitution would corrupt CUSTOM op_attr strings (str.replace
        # matches 'Batch' inside 'MicroBatch').  Weight tensors do not
        # carry a Batch dimension, so the extra pass is unnecessary.

        if args.weight_sharded:
            transformer_qwen3 = ReplicateGraph.apply(
                transformer_qwen3,
                inplace=True,
                old_symbol_map_new_symbol={"fsdp": "dp"},
            )
        else:
            transformer_qwen3 = ReplicateGraph.apply(
                transformer_qwen3, inplace=True, old_symbol_map_new_symbol={"fsdp": 1}
            )

        transformer_qwen3 = GradUpdater.apply(transformer_qwen3, inplace=True)
        spatial_parallel_dims = [dp, tp, spp, ep]

        symbol_map_value[tp] *= symbol_map_value[ep]

        pipeline_tensor_map = _create_pipeline_tensor_map(
            transformer_qwen3.tensors,
            temporal_parallel_dims,
            symbol_map_value,
            num_stacks,
        )

        print("Qwen3-MoE model: Distributing")
        distributed = GraphDistributer.apply(
            transformer_qwen3,
            symbol_map_value,
            spatial_parallel_dims,
            temporal_parallel_dims,
            pipeline_tensor_map,
        )

        if args.print_gpu_vram:
            _print_gpu_vram(
                distributed,
                symbol_map_value,
                mixed_precision=args.mixed_precision,
                header="[Qwen3-MoE] ",
            )

        print("Qwen3-MoE model: Converting Chakra")
        comm_group_file = args.output_name.replace(".%d", "").replace(".et", ".json")
        distributed_chakra = BundledConvertChakra.apply(
            distributed,
            symbol_map_value,
            os.path.join(args.output_dir, comm_group_file),
            mixed_precision=args.mixed_precision,
        )

        from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend import (
            Chakra004Backend as ReadoutBackend,
        )

        distributed_chakra = _postprocess_chakra_graph(distributed_chakra, args, dp)

        print("Qwen3-MoE model: reading out")
        distributed_chakra.readout(generated_filename, backend=ReadoutBackend)

    elif args.model_type == "debug":
        transformer_moe = TensorGraph.load_tensor_graph(
            "./sharding_spreadsheets/module3/tpsp/embedding.csv"
        )
        transformer_moe = ReplicateGraph.apply(
            transformer_moe,
            inplace=True,
            old_symbol_map_new_symbol={
                "Batch": "MicroBatch",
                "Din": "Dvocal",
                "Dout": "Dvocal",
            },
        )

        if args.weight_sharded:
            transformer_moe = ReplicateGraph.apply(
                transformer_moe,
                inplace=True,
                old_symbol_map_new_symbol={"fsdp": "dp"},
            )
        else:
            transformer_moe = ReplicateGraph.apply(
                transformer_moe, inplace=True, old_symbol_map_new_symbol={"fsdp": 1}
            )

        # transformer_moe.visualize("moe")
        # transformer_moe.save_tensor_graph("moe.csv")
        transformer_moe = GradUpdater.apply(transformer_moe, inplace=True)
        spatial_parallel_dims_moe = [dp, tp, spp, ep]

        # moe model
        assert args.pp == 1
        pipeline_tensor_map = {
            "x@0": {pp: 0},
            "w@0": {pp: 0},
            "y@0": {pp: 0},
            "dy@0": {pp: 0},
            "dw@0": {pp: 0},
            "dx@0": {pp: 0},
            "w@1": {pp: 0},
        }

        print("MoE model: Distributing")
        distributed_tensor_graph_moe = GraphDistributer.apply(
            transformer_moe,
            symbol_map_value,
            spatial_parallel_dims_moe,
            temporal_parallel_dims,
            pipeline_tensor_map,
        )

        print("MoE model: Converting Chakra")
        comm_group_file = args.output_name.replace(".%d", "").replace(".et", ".json")
        distributed_chakra_graph_moe = BundledConvertChakra.apply(
            distributed_tensor_graph_moe,
            symbol_map_value,
            os.path.join(args.output_dir, comm_group_file),
        )

        from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend import (
            Chakra004Backend as ReadoutBackend,
        )

        print("MoE model: reading out")
        distributed_chakra_graph_moe = _postprocess_chakra_graph(
            distributed_chakra_graph_moe, args, dp
        )
        distributed_chakra_graph_moe.readout(generated_filename, backend=ReadoutBackend)


if __name__ == "__main__":
    main()
