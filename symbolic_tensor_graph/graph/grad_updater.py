import copy
import typing
import sympy as sp
import os
from collections import defaultdict
from functools import lru_cache
from ..ops import Add, PlaceHolder, Customized, Identical
from ..tensor import Tensor
from ..graph.graph import TensorGraph, BundledHybridGraph, HybridGraph
from ..graph.replicate_graph import ReplicateGraph
from ..graph.connect_graph import ConnectGraph
from ..chakra.node import Node


OPTIMIZED = os.environ.get("STAGE_OPTIMIZED", "1") == "1"


class GradUpdater:
    @classmethod
    def _default_revision_fn(cls, old_replicate):
        return str(int(old_replicate) + 1)

    @classmethod
    def _update_grad(cls, tensor, grad, new_revision_fn):
        print(f"{tensor.name}: {grad.y_shape} @ {grad.y_hidden}")
        updated_tensor = Tensor(create_empty=True)
        updated_tensor.name = tensor.name
        updated_tensor.require_grads = tensor.require_grads
        updated_tensor.x1 = tensor
        updated_tensor.x2 = grad
        updated_tensor.op_type = Add.type_name
        updated_tensor.op_attr = None
        updated_tensor.x1_shape = tensor.y_shape
        updated_tensor.x1_hidden = tensor.y_hidden
        updated_tensor.x2_shape = tensor.y_shape
        updated_tensor.x2_hidden = tensor.y_hidden
        updated_tensor.revision = new_revision_fn(tensor.revision)
        return updated_tensor

    @classmethod
    def apply(cls, graph, new_revision=None, inplace=False):
        if not inplace:
            graph = copy.deepcopy(graph)

        if new_revision is None:
            new_revision = cls._default_revision_fn
        elif isinstance(new_revision, str):
            new_revision = lambda _: new_revision
        elif isinstance(new_revision, typing.Callable):
            pass
        else:
            assert False

        tensor_id_map_tensor = graph.get_tensor_id_map_tensor()
        for tensor_id in tensor_id_map_tensor.keys():
            tensor = tensor_id_map_tensor[tensor_id]
            if tensor.require_grads:
                grad = tensor._grad
                assert grad in graph.out_tensors
                updated_tensor = cls._update_grad(tensor, grad, new_revision)
                graph.out_tensors.remove(grad)
                graph.tensors.append(updated_tensor)
                graph.out_tensors.append(updated_tensor)
        return graph


class FSDPWeightGradManager:
    @classmethod
    def fsdp_weight_distributor(cls, weights, name=None):
        fsdp = sp.symbols("fsdp")

        if name is None:
            name = ""
        reduce_expr = sp.parse_expr("1")
        total_weight_size = 0
        for weight in weights:
            total_weight_size += Tensor.eval_size(weight.y_shape)

        sharded_weight = Tensor(create_empty=True)
        sharded_weight.name = f"{name}_sharded_weight"
        sharded_weight.revision = weights[0].revision
        sharded_weight.require_grads = True
        sharded_weight.op_type = PlaceHolder.type_name
        sharded_weight.x1_shape = [total_weight_size / fsdp]
        sharded_weight.x1_hidden = [reduce_expr]

        assembled_weight = Tensor(create_empty=True)
        assembled_weight.name = f"{name}_assembled_weight"
        assembled_weight.revision = weights[0].revision
        assembled_weight.require_grads = False
        assembled_weight.op_type = Identical.type_name
        assembled_weight.x1 = sharded_weight
        assembled_weight.x1_shape = [total_weight_size]
        assembled_weight.x1_hidden = [reduce_expr]

        for weight in weights:
            assert weight.op_type == PlaceHolder.type_name
            assert weight.require_grads

            weight.op_type = Customized.type_name
            weight.op_attr = "0"
            weight.require_grads = False
            weight.x1 = assembled_weight
            weight.x2_shape = weight.x1_shape
            weight.x2_hidden = weight.x1_hidden
            weight.x1_shape = assembled_weight.x1_shape
            weight.x1_hidden = assembled_weight.x1_hidden
        return sharded_weight, assembled_weight

    @classmethod
    def fsdp_backward_weight_shadow(
        cls,
        tensors,
        sharded_weight,
        assembled_weight,
        weights,
        name=None,
        grad_filter=None,
    ):
        if grad_filter is None:

            def grad_filter(_t):
                return _t.name.split(".")[-1].startswith("d")

        if name is None:
            name = ""
        backward_weights = list()

        assembled_weight_backward = Tensor(create_empty=True)
        assembled_weight_backward.name = f"{name}_assembled_weight_backward"
        assembled_weight_backward.revision = sharded_weight.revision
        assembled_weight_backward.require_grads = False
        assembled_weight_backward.op_type = Identical.type_name
        assembled_weight_backward.x1 = sharded_weight
        assembled_weight_backward.x1_shape = assembled_weight.x1_shape
        assembled_weight_backward.x1_hidden = assembled_weight.x1_hidden

        for weight in weights:
            backward_weight = Tensor(create_empty=True)
            backward_weight.name = f"{name}_{weight.name}_backward"
            backward_weight.revision = weight.revision
            backward_weight.require_grads = False
            backward_weight.op_type = Customized.type_name
            backward_weight.op_attr = "0"
            backward_weight.x1 = assembled_weight_backward
            backward_weight.x1_shape = assembled_weight_backward.x1_shape
            backward_weight.x1_hidden = assembled_weight_backward.x1_hidden
            if os.environ.get("STAGE_MICROBATCH_OPTIMIZE", "1") == "1":
                backward_weight.x2_shape = weight.y_shape
                backward_weight.x2_hidden = weight.y_hidden
            else:
                backward_weight.x2_shape = weight.x1_shape
                backward_weight.x2_hidden = weight.x1_hidden
            backward_weights.append(backward_weight)

        weight_map_backward_weight = dict()
        for weight, backward_weight in zip(weights, backward_weights):
            weight_map_backward_weight[weight] = backward_weight

        for tensor in filter(grad_filter, tensors):
            if tensor.x1 in weight_map_backward_weight:
                tensor.x1 = weight_map_backward_weight[tensor.x1]
            if tensor.x2 in weight_map_backward_weight:
                tensor.x2 = weight_map_backward_weight[tensor.x2]
        return backward_weights, assembled_weight_backward

    @classmethod
    def fsdp_grad_gatherer(cls, grads, assembled_weight, name=None):
        if name is None:
            name = ""
        reduce_expr = sp.parse_expr("1/(cp*dp)")
        assembled_grad = Tensor(create_empty=True)
        assembled_grad.name = f"{name}_assembled_grad"
        assembled_grad.revision = grads[0].revision
        assembled_grad.require_grads = False
        assembled_grad.op_type = Customized.type_name
        assembled_grad.op_attr = "0"
        assembled_grad.x1_shape = grads[0].y_shape
        assembled_grad.x1_hidden = grads[0].y_hidden
        assembled_grad.x2_shape = assembled_weight.y_shape
        assembled_grad.x2_hidden = grads[0].y_hidden
        assembled_grad.x1 = grads[0]
        assembled_grad.extra_attr["data_deps"] = list()
        for i, grad in enumerate(grads):
            if i == 0:
                continue
            assembled_grad.extra_attr["data_deps"].append(grad)

        sharded_grad = Tensor(create_empty=True)
        sharded_grad.name = f"{name}_sharded_grad"
        sharded_grad.revision = grads[0].revision
        sharded_grad.require_grads = False
        sharded_grad.op_type = Identical.type_name
        sharded_grad.x1 = assembled_grad

        sharded_weight = assembled_weight.x1
        sharded_grad.x1_shape = sharded_weight.y_shape
        sharded_grad.x1_hidden = sharded_weight.y_hidden
        sharded_grad.grad_of = sharded_weight
        sharded_weight._grad = sharded_grad

        return sharded_grad, assembled_grad

    @classmethod
    def apply(cls, graph, inplace=False):
        if not inplace:
            graph = copy.deepcopy(graph)
        tensor_id_map_tensor = graph.get_tensor_id_map_tensor()
        weights = list()
        grads = list()
        for tensor in tensor_id_map_tensor.values():
            if tensor.op_type == PlaceHolder.type_name and tensor.require_grads:
                weights.append(tensor)
                grads.append(tensor._grad)
                graph.in_tensors.remove(tensor)
                graph.out_tensors.remove(tensor._grad)
        sharded_weight, assembled_weight = cls.fsdp_weight_distributor(weights)
        sharded_grad, assembled_grad = cls.fsdp_grad_gatherer(grads, assembled_weight)
        graph.tensors.append(sharded_weight)
        graph.tensors.append(assembled_weight)
        graph.tensors.append(sharded_grad)
        graph.tensors.append(assembled_grad)
        graph.in_tensors.append(sharded_weight)
        graph.out_tensors.append(sharded_grad)
        backward_weights, assembled_weight_backward = cls.fsdp_backward_weight_shadow(
            graph.tensors, sharded_weight, assembled_weight, weights
        )
        graph.tensors.extend(backward_weights)
        graph.tensors.append(assembled_weight_backward)

        return graph


class MicroBatchReplicator:
    @classmethod
    def get_weights_grads_others(cls, graph):
        weights = list()
        grads = list()
        others = list()
        for tensor in graph.tensors:
            if tensor.op_type == PlaceHolder.type_name and tensor.require_grads:
                weights.append(tensor)
                grads.append(tensor._grad)
            else:
                others.append(tensor)
        for grad in grads:
            others.remove(grad)
        return weights, grads, others

    @classmethod
    def _should_merge_before_sync(cls, grad):
        if grad.x1 is None:
            return False
        if grad.x1_shape is None or grad.x1_hidden is None:
            return False
        return grad.x1_shape != grad.x1.y_shape or grad.x1_hidden != grad.x1.y_hidden

    @classmethod
    def _create_merged_grad(cls, name, revision, grads):
        assert len(grads) > 0
        merged_grad = Tensor(create_empty=True)
        merged_grad.name = name
        merged_grad.revision = revision
        merged_grad.require_grads = False
        merged_grad.op_type = Customized.type_name
        merged_grad.op_attr = str(Tensor.eval_size(grads[0].y_shape))
        merged_grad.x1 = grads[0]
        merged_grad.x1_shape = grads[0].y_shape
        merged_grad.x1_hidden = grads[0].y_hidden
        merged_grad.x2_shape = grads[0].y_shape
        merged_grad.x2_hidden = grads[0].y_hidden
        merged_grad.extra_attr["data_deps"] = list()
        for grad in grads[1:]:
            merged_grad.extra_attr["data_deps"].append(grad)
        return merged_grad

    @classmethod
    def apply(cls, graph, symbol_map_value, inplace=False):
        # raise NotImplementedError("Too slow, use the postprocess instead")
        batch, microbatch = sp.symbols("Batch MicroBatch")
        assert microbatch in symbol_map_value
        assert batch in symbol_map_value
        num_batches = symbol_map_value[batch] / symbol_map_value[microbatch]
        assert int(num_batches) == num_batches
        num_batches = int(num_batches)

        if not inplace:
            graph = copy.deepcopy(graph)
        weights, grads, _ = cls.get_weights_grads_others(graph)

        microbatch_graphs = list()

        for i in range(num_batches):
            microbatch_graph = ReplicateGraph.apply(
                graph, f"mb{i}.%s", old_symbol_map_new_symbol={batch: microbatch}
            )
            microbatch_graphs.append(microbatch_graph)

        merged_graph = ConnectGraph.apply(microbatch_graphs, dict())

        new_weight_map_old_weight = dict()
        old_grad_map_new_grads = dict()
        for tensor in merged_graph.tensors:
            for weight in weights:
                if tensor.name[tensor.name.find(".") + 1 :] == weight.name:
                    new_weight_map_old_weight[tensor] = weight
                    break
            for grad in grads:
                if tensor.name[tensor.name.find(".") + 1 :] == grad.name:
                    if not grad in old_grad_map_new_grads:
                        old_grad_map_new_grads[grad] = list()
                    old_grad_map_new_grads[grad].append(tensor)
                    break

        for tensor in merged_graph.tensors:
            if tensor.x1 in new_weight_map_old_weight:
                tensor.x1 = new_weight_map_old_weight[tensor.x1]
            if tensor.x2 in new_weight_map_old_weight:
                tensor.x2 = new_weight_map_old_weight[tensor.x2]

        old_grad_map_merged_grad = dict()
        old_grad_map_wrapped_grads = dict()
        for old_grad in old_grad_map_new_grads:
            merge_inputs = old_grad_map_new_grads[old_grad]
            merge_name = old_grad.name
            merge_revision = old_grad.revision
            old_grad_map_wrapped_grads[old_grad] = list()
            if cls._should_merge_before_sync(old_grad):
                # Keep per-microbatch gradients local, then let the step-level
                # weight update trigger a single DP synchronization.
                merge_inputs = [new_grad.x1 for new_grad in old_grad_map_new_grads[old_grad]]
                merge_name = old_grad.x1.name
                merge_revision = old_grad.x1.revision
                old_grad_map_wrapped_grads[old_grad] = old_grad_map_new_grads[old_grad]

            merged_grad = cls._create_merged_grad(
                merge_name, merge_revision, merge_inputs
            )
            old_grad_map_merged_grad[old_grad] = merged_grad
            merged_graph.tensors.append(merged_grad)
            merged_graph.out_tensors.append(merged_grad)
            for new_grad in old_grad_map_new_grads[old_grad]:
                merged_graph.out_tensors.remove(new_grad)
            for wrapped_grad in old_grad_map_wrapped_grads[old_grad]:
                merged_graph.tensors.remove(wrapped_grad)

        for old_grad in old_grad_map_new_grads:
            for new_grad in old_grad_map_new_grads[old_grad]:
                new_weight = new_grad.grad_of
                old_weight = new_weight_map_old_weight[new_weight]
                if new_grad in merged_graph.tensors:
                    new_grad.grad_of = old_weight
                old_weight._grad = old_grad_map_merged_grad[old_grad]
                merged_graph.tensors.remove(new_weight)
                merged_graph.in_tensors.remove(new_weight)
        for old_weight in weights:
            merged_graph.in_tensors.append(old_weight)
            merged_graph.tensors.append(old_weight)

        merged_graph.sanity_check()
        return merged_graph


class MicroBatchReplicatorPostProcess:
    OFFSET = 1000000000

    @classmethod
    def find_weights_grads(cls, graph: HybridGraph):
        weights_map_grads = dict()
        for tensor in graph.tensors:
            if tensor.op_type == PlaceHolder.type_name and tensor.require_grads:
                weights_map_grads[tensor] = tensor._grad
        return weights_map_grads

    @classmethod
    def replicate_micro_batches(cls, graph: HybridGraph, num_micro_batches):
        if num_micro_batches == 1:
            return graph
        # this is not accurate, but works.
        for tensor in graph.tensor_map_nodes.keys():
            nodes_this_tensor = graph.tensor_map_nodes[tensor]
            old_keys = list(nodes_this_tensor.keys())
            for mb in range(num_micro_batches):
                if mb == 0:
                    continue
                for key in old_keys:
                    old_node = nodes_this_tensor[key]
                    new_key = f"mb{mb}_{key}"
                    new_node = copy.deepcopy(old_node)
                    new_node.name = f"mb{mb}.{old_node.name}"
                    new_node.id = old_node.id + cls.OFFSET * mb
                    data_deps = old_node.data_deps
                    new_node.data_deps = list()
                    for data_dep in data_deps:
                        new_node.data_deps.append(data_dep + cls.OFFSET * mb)
                    ctrl_deps = old_node.ctrl_deps
                    new_node.ctrl_deps = list()
                    for ctrl_dep in ctrl_deps:
                        new_node.ctrl_deps.append(ctrl_dep + cls.OFFSET * mb)
                    nodes_this_tensor[new_key] = new_node
            graph.tensor_map_nodes[tensor] = nodes_this_tensor

    @classmethod
    def apply(cls, bundled_graph: BundledHybridGraph, num_micro_batches, inplace=True):
        if not OPTIMIZED:
            return cls.apply_no_optimize(bundled_graph, num_micro_batches, inplace)
        assert inplace
        print("Replicating micro batches")
        for readable_rank in bundled_graph.graphs.keys():
            # optimize
            is_zero_rank = True
            for sym, rank in readable_rank:
                if sym in bundled_graph.spatial_parallel_dims and rank != 0:
                    is_zero_rank = False
                    break
            if not is_zero_rank:
                continue
            # print(f"Rank {readable_rank}")
            hybrid_graph = bundled_graph.graphs[readable_rank]
            cls.replicate_micro_batches(hybrid_graph, num_micro_batches)
        print("Replicate micro batches done")
        return bundled_graph

    @classmethod
    def apply_no_optimize(
        cls, bundled_graph: BundledHybridGraph, num_micro_batches, inplace=True
    ):
        assert inplace
        print("Replicating micro batches")
        for readable_rank in bundled_graph.graphs.keys():
            # print(f"Rank {readable_rank}")
            hybrid_graph = bundled_graph.graphs[readable_rank]
            cls.replicate_micro_batches(hybrid_graph, num_micro_batches)
        print("Replicate micro batches done")
        return bundled_graph


class LocalSGDIterationPostProcess:
    @classmethod
    def _iteration_key(cls, base_key, iteration):
        if iteration == 0:
            return base_key
        return f"iter{iteration}_{base_key}"

    @classmethod
    def _iteration_name(cls, base_name, iteration):
        if iteration == 0:
            return base_name
        return f"iter{iteration}.{base_name}"

    @classmethod
    def _is_sync_iteration(cls, iteration, sync_interval):
        return sync_interval == 1 or ((iteration + 1) % sync_interval == 0)

    @classmethod
    def _get_comm_tag_bounds(cls, graph: HybridGraph):
        min_comm_tag = None
        max_comm_tag = None
        for tensor, nodes_this_tensor in graph.tensor_map_nodes.items():
            for key, node in nodes_this_tensor.items():
                if node.node_type in {
                    Node.NodeType.COMM_SEND_NODE,
                    Node.NodeType.COMM_RECV_NODE,
                } and hasattr(node, "comm_tag"):
                    if min_comm_tag is None or node.comm_tag < min_comm_tag:
                        min_comm_tag = node.comm_tag
                    if max_comm_tag is None or node.comm_tag > max_comm_tag:
                        max_comm_tag = node.comm_tag
        return min_comm_tag, max_comm_tag

    @classmethod
    def _get_comm_tag_stride(cls, graph: HybridGraph):
        min_comm_tag, max_comm_tag = cls._get_comm_tag_bounds(graph)
        if min_comm_tag is not None and max_comm_tag is not None:
            return max_comm_tag - min_comm_tag + 1
        return 0

    @classmethod
    def _get_bundled_comm_tag_stride(cls, bundled_graph):
        min_comm_tag = None
        max_comm_tag = None
        for graph in bundled_graph.graphs.values():
            graph_min_comm_tag, graph_max_comm_tag = cls._get_comm_tag_bounds(graph)
            if graph_min_comm_tag is None or graph_max_comm_tag is None:
                continue
            if min_comm_tag is None or graph_min_comm_tag < min_comm_tag:
                min_comm_tag = graph_min_comm_tag
            if max_comm_tag is None or graph_max_comm_tag > max_comm_tag:
                max_comm_tag = graph_max_comm_tag
        if min_comm_tag is not None and max_comm_tag is not None:
            return max_comm_tag - min_comm_tag + 1
        return 0

    @classmethod
    def _freeze_template_nodes(cls, graph: HybridGraph):
        template_nodes = dict()
        template_keys = dict()
        max_node_id = 0
        for tensor, nodes_this_tensor in graph.tensor_map_nodes.items():
            template_keys[tensor] = list(nodes_this_tensor.keys())
            template_nodes[tensor] = dict()
            for key, node in nodes_this_tensor.items():
                template_nodes[tensor][key] = copy.deepcopy(node)
                max_node_id = max(max_node_id, node.id)
        return template_nodes, template_keys, max_node_id + 1

    @classmethod
    def _clone_node_for_iteration(cls, node, iteration, node_id_stride, comm_tag_stride):
        if iteration == 0:
            return node
        new_node = copy.deepcopy(node)
        new_node.name = cls._iteration_name(node.name, iteration)
        new_node.id = node.id + node_id_stride * iteration
        new_node.data_deps = [dep + node_id_stride * iteration for dep in node.data_deps]
        new_node.ctrl_deps = [dep + node_id_stride * iteration for dep in node.ctrl_deps]
        if (
            comm_tag_stride > 0
            and node.node_type
            in {Node.NodeType.COMM_SEND_NODE, Node.NodeType.COMM_RECV_NODE}
            and hasattr(new_node, "comm_tag")
        ):
            new_node.comm_tag += comm_tag_stride * iteration
        return new_node

    @classmethod
    def _get_iteration_node_refs(cls, graph: HybridGraph, template_keys, iteration):
        node_refs = list()
        iteration_nodes = dict()
        for tensor, base_keys in template_keys.items():
            nodes_this_tensor = graph.tensor_map_nodes[tensor]
            for base_key in base_keys:
                iteration_key = cls._iteration_key(base_key, iteration)
                if iteration_key not in nodes_this_tensor:
                    continue
                node = nodes_this_tensor[iteration_key]
                node_refs.append((tensor, iteration_key, node))
                iteration_nodes[node.id] = node
        return node_refs, iteration_nodes

    @classmethod
    def _remove_non_sync_dp_allreduces(
        cls, graph: HybridGraph, template_keys, iteration, dp_symbol
    ):
        node_refs, iteration_nodes = cls._get_iteration_node_refs(
            graph, template_keys, iteration
        )
        removed_node_ids = {
            node.id
            for _, _, node in node_refs
            if node.node_type == Node.NodeType.COLL_COMM_NODE
            and node.comm_type == Node.CollectiveType.ALL_REDUCE
            and getattr(node, "_comm_meta_data", None) is not None
            and node._comm_meta_data[3] == dp_symbol
        }
        if len(removed_node_ids) == 0:
            return iteration_nodes

        @lru_cache(maxsize=None)
        def resolve_surviving_parents(node_id):
            if node_id not in removed_node_ids:
                return (node_id,)
            surviving_parents = list()
            for parent_id in iteration_nodes[node_id].data_deps:
                for resolved_parent_id in resolve_surviving_parents(parent_id):
                    if resolved_parent_id not in surviving_parents:
                        surviving_parents.append(resolved_parent_id)
            return tuple(surviving_parents)

        for node_id, node in iteration_nodes.items():
            if node_id in removed_node_ids:
                continue
            new_data_deps = list()
            for parent_id in node.data_deps:
                for resolved_parent_id in resolve_surviving_parents(parent_id):
                    if resolved_parent_id not in new_data_deps:
                        new_data_deps.append(resolved_parent_id)
            node.data_deps = new_data_deps

        for tensor, iteration_key, node in node_refs:
            if node.id in removed_node_ids:
                del graph.tensor_map_nodes[tensor][iteration_key]
                del iteration_nodes[node.id]

        return iteration_nodes

    @classmethod
    def _get_iteration_boundary_nodes(cls, iteration_nodes):
        parent_to_children = defaultdict(set)
        for node in iteration_nodes.values():
            for parent_id in node.data_deps:
                if parent_id in iteration_nodes:
                    parent_to_children[parent_id].add(node.id)

        source_node_ids = list()
        sink_node_ids = list()
        for node in iteration_nodes.values():
            has_in_iteration_parent = any(
                parent_id in iteration_nodes for parent_id in node.data_deps
            )
            if not has_in_iteration_parent:
                source_node_ids.append(node.id)
            if len(parent_to_children[node.id]) == 0:
                sink_node_ids.append(node.id)

        return source_node_ids, sink_node_ids

    @classmethod
    def _create_iteration_barrier(cls, barrier_id, prev_iteration, next_iteration, deps):
        barrier = Node()
        barrier.node_type = Node.NodeType.COMP_NODE
        barrier.name = f"iter{prev_iteration}_to_iter{next_iteration}_BARRIER"
        barrier.id = barrier_id
        # Keep the barrier as a real executable node so Astra-Sim advances
        # dependents via the normal callback path instead of treating it as
        # an invalid zero-size comp node.
        barrier.num_ops = 16384
        barrier.tensor_size = 16384
        barrier.y_tensor_size = 16384
        barrier.op_type = "barrier"
        barrier.data_deps = list(deps)
        barrier.ctrl_deps = list()
        barrier.inputs = list()
        barrier.outputs = list()
        return barrier

    @classmethod
    def replicate_iterations(
        cls,
        graph: HybridGraph,
        num_iterations,
        sync_interval,
        dp_symbol,
        comm_tag_stride=None,
        inplace=True,
    ):
        if not inplace:
            graph = copy.deepcopy(graph)
        if num_iterations < 1:
            raise ValueError("num_iterations must be at least 1")
        if sync_interval < 1:
            raise ValueError("sync_interval must be at least 1")
        if num_iterations == 1 and sync_interval == 1:
            return graph
        if len(graph.tensor_map_nodes) == 0:
            return graph
        if comm_tag_stride is None:
            comm_tag_stride = cls._get_comm_tag_stride(graph)

        (
            template_nodes,
            template_keys,
            node_id_stride,
        ) = cls._freeze_template_nodes(graph)
        anchor_tensor = next(iter(graph.tensor_map_nodes.keys()))

        for iteration in range(1, num_iterations):
            for tensor, base_keys in template_keys.items():
                nodes_this_tensor = graph.tensor_map_nodes[tensor]
                for base_key in base_keys:
                    iteration_key = cls._iteration_key(base_key, iteration)
                    nodes_this_tensor[iteration_key] = cls._clone_node_for_iteration(
                        template_nodes[tensor][base_key],
                        iteration,
                        node_id_stride,
                        comm_tag_stride,
                    )

        iteration_sources = dict()
        iteration_sinks = dict()
        for iteration in range(num_iterations):
            if not cls._is_sync_iteration(iteration, sync_interval):
                iteration_nodes = cls._remove_non_sync_dp_allreduces(
                    graph, template_keys, iteration, dp_symbol
                )
            else:
                _, iteration_nodes = cls._get_iteration_node_refs(
                    graph, template_keys, iteration
                )
            sources, sinks = cls._get_iteration_boundary_nodes(iteration_nodes)
            iteration_sources[iteration] = sources
            iteration_sinks[iteration] = sinks

        for iteration in range(num_iterations - 1):
            if len(iteration_sinks[iteration]) == 0:
                continue
            if len(iteration_sources[iteration + 1]) == 0:
                continue
            barrier_id = node_id_stride * (iteration + 1)
            barrier = cls._create_iteration_barrier(
                barrier_id,
                iteration,
                iteration + 1,
                iteration_sinks[iteration],
            )
            graph.tensor_map_nodes[anchor_tensor][
                f"local_sgd_barrier_{iteration}"
            ] = barrier

            _, next_iteration_nodes = cls._get_iteration_node_refs(
                graph, template_keys, iteration + 1
            )
            for source_id in iteration_sources[iteration + 1]:
                source_node = next_iteration_nodes[source_id]
                if barrier_id not in source_node.data_deps:
                    source_node.data_deps.append(barrier_id)

        return graph

    @classmethod
    def apply(
        cls,
        bundled_graph: BundledHybridGraph,
        num_iterations,
        sync_interval,
        dp_symbol,
        inplace=True,
    ):
        if not inplace:
            bundled_graph = copy.deepcopy(bundled_graph)
        if num_iterations == 1 and sync_interval == 1:
            return bundled_graph
        print("Applying LocalSGD iteration postprocess")
        comm_tag_stride = cls._get_bundled_comm_tag_stride(bundled_graph)
        for readable_rank in bundled_graph.graphs.keys():
            if OPTIMIZED:
                is_zero_rank = True
                for sym, rank in readable_rank:
                    if sym in bundled_graph.spatial_parallel_dims and rank != 0:
                        is_zero_rank = False
                        break
                if not is_zero_rank:
                    continue
            hybrid_graph = bundled_graph.graphs[readable_rank]
            cls.replicate_iterations(
                hybrid_graph,
                num_iterations=num_iterations,
                sync_interval=sync_interval,
                dp_symbol=dp_symbol,
                comm_tag_stride=comm_tag_stride,
                inplace=True,
            )
        print("LocalSGD iteration postprocess done")
        return bundled_graph
