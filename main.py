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
from symbolic_tensor_graph.graph.pipeline_schedule import PipelineScheduleInjector
from symbolic_tensor_graph.graph.activation_recompute import ActivationRecomputePostProcess
import re
from symbolic_tensor_graph.vram_counting import _print_gpu_vram

mixprecision = False


def str_to_bool(v):
    # Convert "true" to True and "false" to False
    return v.lower() in ("true", "t", "1", "yes", "y")


def _build_chunk_cumulative_bounds(num_stacks, num_chunks):
    """Split num_stacks transformer blocks into num_chunks chunks with remainder
    distributed to early chunks. Returns cumulative upper bounds (length=num_chunks)."""
    sizes = [num_stacks // num_chunks] * num_chunks
    for i in range(num_stacks % num_chunks):
        sizes[i] += 1
    cumulative = []
    acc = 0
    for s in sizes:
        acc += s
        cumulative.append(acc)
    return cumulative


def _block_idx_to_device(block_idx, cumulative, range_):
    """Given cumulative chunk bounds and pipeline-parallel degree (range_),
    return the device (pp rank) that owns the given transformer block.

    With num_chunks = len(cumulative) = virtual_stages * range_, consecutive
    chunks are dealt round-robin to devices: chunk j -> device j % range_.
    This reproduces contiguous-stage mapping when virtual_stages == 1."""
    chunk_idx = next(i for i, up in enumerate(cumulative) if block_idx < up)
    return chunk_idx % range_


def _create_pipeline_tensor_map_mix_precision(
    _tensors, _temporal_parallel_dims, _symbol_map_value, num_stacks,
    virtual_stages=1,
):
    _tensor_map = dict()
    assert len(_temporal_parallel_dims) == 1
    parallel_dim = _temporal_parallel_dims[0]
    range_ = _symbol_map_value[parallel_dim]
    assert virtual_stages >= 1
    num_chunks = virtual_stages * range_
    assert num_stacks >= num_chunks, (
        f"num_stacks ({num_stacks}) must be >= virtual_stages * pp "
        f"({virtual_stages} * {range_} = {num_chunks})"
    )
    cumulative = _build_chunk_cumulative_bounds(num_stacks, num_chunks)
    # Last transformer block's device — used for out_emb / loss so they stay
    # on the same device as the final layer regardless of interleaving.
    last_device = _block_idx_to_device(num_stacks - 1, cumulative, range_)

    block_to_chunk_local = {}
    for block_idx in range(num_stacks):
        chunk_idx = next(i for i, up in enumerate(cumulative) if block_idx < up)
        block_to_chunk_local[block_idx] = chunk_idx // range_

    for tensor in _tensors:
        tid = tensor.id
        m = re.search(r"transformer\.(\d+)", tid)
        if m:
            block_idx = int(m.group(1))
            device = _block_idx_to_device(block_idx, cumulative, range_)
            _tensor_map[tid] = {parallel_dim: device}
            continue

        if "in_emb" in tid:
            _tensor_map[tid] = {parallel_dim: 0}
        elif "out_emb" in tid or "loss" in tid:
            _tensor_map[tid] = {parallel_dim: last_device}
        else:
            raise ValueError(f"Unrecognized tensor id for pipeline mapping: {tid}")

    return _tensor_map, block_to_chunk_local


def _create_pipeline_tensor_map(
    _tensors, _temporal_parallel_dims, _symbol_map_value, num_stacks,
    virtual_stages=1,
):
    if mixprecision:
        return _create_pipeline_tensor_map_mix_precision(
            _tensors, _temporal_parallel_dims, _symbol_map_value, num_stacks,
            virtual_stages=virtual_stages,
        )
    _tensor_map = dict()
    assert len(_temporal_parallel_dims) == 1
    parallel_dim = _temporal_parallel_dims[0]
    range_ = _symbol_map_value[parallel_dim]
    assert virtual_stages >= 1
    num_chunks = virtual_stages * range_
    assert num_stacks >= num_chunks, (
        f"num_stacks ({num_stacks}) must be >= virtual_stages * pp "
        f"({virtual_stages} * {range_} = {num_chunks})"
    )
    cumulative = _build_chunk_cumulative_bounds(num_stacks, num_chunks)
    last_device = _block_idx_to_device(num_stacks - 1, cumulative, range_)

    block_to_chunk_local = {}
    for block_idx in range(num_stacks):
        chunk_idx = next(i for i, up in enumerate(cumulative) if block_idx < up)
        block_to_chunk_local[block_idx] = chunk_idx // range_

    for tensor in _tensors:
        found = False
        for num_stack in range(num_stacks):
            if f"transformer.{num_stack}." in tensor.id:
                device = _block_idx_to_device(num_stack, cumulative, range_)
                _tensor_map[tensor.id] = {parallel_dim: device}
                found = True
                break
        if found:
            pass
        elif "in_emb" in tensor.id:
            _tensor_map[tensor.id] = {parallel_dim: 0}
        elif "out_emb" in tensor.id:
            _tensor_map[tensor.id] = {parallel_dim: last_device}
        elif "loss" in tensor.id:
            _tensor_map[tensor.id] = {parallel_dim: last_device}
        else:
            assert False, tensor.name
    return _tensor_map, block_to_chunk_local


def _postprocess_chakra_graph(chakra_graph, args, dp, pp, block_to_chunk_local=None):
    if os.environ.get("STAGE_MICROBATCH_OPTIMIZE", "0") != "0":
        chakra_graph = MicroBatchReplicatorPostProcess.apply(
            chakra_graph, args.batch // (args.micro_batch * args.dp)
        )
    # Inflate backward compute to cover the extra forward pass under
    # activation recomputation (P1-B). Run before schedule injection so that
    # ctrl_deps target the already-inflated nodes.
    if args.activation_recompute:
        chakra_graph = ActivationRecomputePostProcess.apply(chakra_graph)
    # Inject explicit pipeline-schedule ctrl_deps (no-op for 'natural').
    if args.pipeline_schedule != "natural" and args.pp > 1:
        num_mb = args.batch // (args.micro_batch * args.dp)
        chakra_graph = PipelineScheduleInjector.apply(
            chakra_graph,
            schedule=args.pipeline_schedule,
            num_micro_batches=num_mb,
            pipeline_parallel_size=args.pp,
            virtual_stages=args.pipeline_virtual_stages,
            pp_dim_symbol=pp,
            block_to_chunk_local=block_to_chunk_local,
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
        "--pipeline_virtual_stages",
        type=int,
        default=1,
        required=False,
        help="Number of virtual pipeline stages (model chunks) per device. "
             "v=1 is the standard contiguous mapping; v>1 enables Megatron-LM "
             "interleaved pipeline where each device holds v non-contiguous "
             "chunks. Requires num_stacks %% (v * pp) == 0.",
    )
    parser.add_argument(
        "--pipeline_schedule",
        type=str,
        default="natural",
        required=False,
        choices=list(PipelineScheduleInjector.VALID_SCHEDULES),
        help="Pipeline scheduling strategy. 'natural' leaves execution "
             "dependency-driven (legacy behavior). 'gpipe' forces all "
             "forward passes before any backward pass. '1f1b' uses Megatron/"
             "PipeDream-Flush warmup-steady-cooldown ordering. "
             "'1f1b-interleaved' adds chunk-level interleaving for v>1.",
    )
    parser.add_argument(
        "--scatter_gather_optimization",
        type=str_to_bool,
        default=False,
        required=False,
        help="Enable Megatron-LM scatter/gather optimization across pipeline "
             "stage boundaries (§4.1). Each TP rank sends only a 1/t slice "
             "across the inter-node link; the receiver reconstructs the full "
             "tensor via an intra-TP all-gather. Only active when tp > 1.",
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
    parser.add_argument("--batch", type=int, default=64, required=False,
                        help="global batch size across all DP ranks")
    parser.add_argument("--micro_batch", type=int, default=-1, required=False,
                        help="per-rank micro-batch size (Megatron convention). "
                             "Number of micro-batches per iteration is "
                             "batch // (micro_batch * dp). Default -1 means one "
                             "micro-batch per rank (no gradient accumulation).")
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

    args = parser.parse_args()
    if args.num_iterations < 1:
        raise ValueError("--num_iterations must be at least 1")
    if args.dp_local_sgd_interval < 1:
        raise ValueError("--dp_local_sgd_interval must be at least 1")
    if args.pipeline_virtual_stages < 1:
        raise ValueError("--pipeline_virtual_stages must be at least 1")
    # Divisibility is only strictly required for clean interleaved scheduling.
    # v=1 (contiguous mapping) tolerates remainders, matching legacy behavior.
    if args.pipeline_virtual_stages > 1 and \
       args.num_stacks % (args.pipeline_virtual_stages * args.pp) != 0:
        raise ValueError(
            f"--num_stacks ({args.num_stacks}) must be divisible by "
            f"--pipeline_virtual_stages ({args.pipeline_virtual_stages}) * "
            f"--pp ({args.pp}) = "
            f"{args.pipeline_virtual_stages * args.pp} "
            "when using interleaved pipeline (v > 1)"
        )
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
    # --micro_batch is per-rank (Megatron convention). One iteration produces
    # batch // (micro_batch * dp) micro-batches. Default -1 -> no grad accum.
    if args.micro_batch == -1:
        if args.batch % args.dp != 0:
            raise ValueError(
                f"--batch ({args.batch}) must be divisible by --dp ({args.dp}) "
                "when --micro_batch is left at its default"
            )
        args.micro_batch = args.batch // args.dp
    if args.batch % (args.micro_batch * args.dp) != 0:
        raise ValueError(
            f"--batch ({args.batch}) must be divisible by "
            f"--micro_batch ({args.micro_batch}) * --dp ({args.dp})"
        )
    # The graph still uses the symbolic shape `Batch/dp` per rank. After
    # MicroBatchReplicator substitutes Batch -> MicroBatch, each replicated
    # micro-batch graph evaluates to args.micro_batch samples per rank, which
    # requires the symbolic MicroBatch value to be (per-rank micro-batch * dp).
    symbol_map_value = {
        Dvocal: args.dvocal,
        Dmodel: args.dmodel,
        Dff: args.dff,
        Batch: args.batch,
        MicroBatch: args.micro_batch * args.dp,
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
        pipeline_tensor_map, block_to_chunk_local = _create_pipeline_tensor_map(
            transformer_dense.tensors,
            temporal_parallel_dims,
            symbol_map_value,
            num_stacks,
            virtual_stages=args.pipeline_virtual_stages,
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
                activation_recompute=args.activation_recompute,
            )

        print("Dense model: Converting Chakra")
        comm_group_file = args.output_name.replace(".%d", "").replace(".et", ".json")
        distributed_chakra_graph_dense = BundledConvertChakra.apply(
            distributed_tensor_graph_dense,
            symbol_map_value,
            os.path.join(args.output_dir, comm_group_file),
            mixed_precision=args.mixed_precision,
            scatter_gather_optimization=args.scatter_gather_optimization,
            pp_dim_symbol=pp,
            tp_dim_symbol=tp,
        )

        from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend import (
            Chakra004Backend as ReadoutBackend,
        )

        distributed_chakra_graph_dense = _postprocess_chakra_graph(
            distributed_chakra_graph_dense, args, dp, pp,
            block_to_chunk_local=block_to_chunk_local,
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
        pipeline_tensor_map, block_to_chunk_local = _create_pipeline_tensor_map(
            transformer_dense.tensors,
            temporal_parallel_dims,
            symbol_map_value,
            num_stacks,
            virtual_stages=args.pipeline_virtual_stages,
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
            scatter_gather_optimization=args.scatter_gather_optimization,
            pp_dim_symbol=pp,
            tp_dim_symbol=tp,
        )

        from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend import (
            Chakra004Backend as ReadoutBackend,
        )

        print("Dense model: reading out")
        distributed_chakra_graph_dense = _postprocess_chakra_graph(
            distributed_chakra_graph_dense, args, dp, pp,
            block_to_chunk_local=block_to_chunk_local,
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
        pipeline_tensor_map, block_to_chunk_local = _create_pipeline_tensor_map(
            transformer_moe.tensors,
            temporal_parallel_dims,
            symbol_map_value,
            num_stacks,
            virtual_stages=args.pipeline_virtual_stages,
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
                activation_recompute=args.activation_recompute,
            )

        print("MoE model: Converting Chakra")
        comm_group_file = args.output_name.replace(".%d", "").replace(".et", ".json")
        distributed_chakra_graph_moe = BundledConvertChakra.apply(
            distributed_tensor_graph_moe,
            symbol_map_value,
            os.path.join(args.output_dir, comm_group_file),
            mixed_precision=args.mixed_precision,
            scatter_gather_optimization=args.scatter_gather_optimization,
            pp_dim_symbol=pp,
            tp_dim_symbol=tp,
        )

        from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend import (
            Chakra004Backend as ReadoutBackend,
        )

        print("MoE model: reading out")
        distributed_chakra_graph_moe = _postprocess_chakra_graph(
            distributed_chakra_graph_moe, args, dp, pp,
            block_to_chunk_local=block_to_chunk_local,
        )
        distributed_chakra_graph_moe.readout(generated_filename, backend=ReadoutBackend)

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
            scatter_gather_optimization=args.scatter_gather_optimization,
            pp_dim_symbol=pp,
            tp_dim_symbol=tp,
        )

        from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend import (
            Chakra004Backend as ReadoutBackend,
        )

        print("MoE model: reading out")
        distributed_chakra_graph_moe = _postprocess_chakra_graph(
            distributed_chakra_graph_moe, args, dp, pp
        )
        distributed_chakra_graph_moe.readout(generated_filename, backend=ReadoutBackend)


if __name__ == "__main__":
    main()
