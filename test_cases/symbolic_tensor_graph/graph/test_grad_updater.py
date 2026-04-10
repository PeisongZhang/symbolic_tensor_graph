import unittest
import sympy as sp


class TestGradUpdater(unittest.TestCase):
    def _make_placeholder(self, name, require_grads, shape, hidden):
        from symbolic_tensor_graph.tensor import Tensor
        from symbolic_tensor_graph.ops import PlaceHolder

        tensor = Tensor(create_empty=True)
        tensor.name = name
        tensor.revision = "0"
        tensor.require_grads = require_grads
        tensor.op_type = PlaceHolder.type_name
        tensor.op_attr = None
        tensor.x1_shape = shape
        tensor.x1_hidden = hidden
        return tensor

    def _make_weight_sync_graph(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph
        from symbolic_tensor_graph.ops import Identical

        Batch, MicroBatch, dp = sp.symbols("Batch MicroBatch dp")

        weight = self._make_placeholder("w", True, [4], [1])
        local_grad = self._make_placeholder("assembled_grad", False, [4], [Batch / dp])

        synced_grad = self._make_placeholder("sharded_grad", False, [4], [1])
        synced_grad.op_type = Identical.type_name
        synced_grad.x1 = local_grad
        synced_grad.grad_of = weight
        weight._grad = synced_grad

        graph = TensorGraph(
            [weight, local_grad, synced_grad],
            in_tensors=[weight, local_grad],
            out_tensors=[synced_grad],
        )
        symbol_map = {
            Batch: 8,
            MicroBatch: 2,
            dp: 4,
        }
        return graph, symbol_map, dp

    def _make_point_to_point_graph(self, send_dst=1, recv_src=0, base_tag=7, extra_tag=None):
        from symbolic_tensor_graph.graph.graph import HybridGraph
        from symbolic_tensor_graph.chakra.node import Node

        def make_pair(name, tag, start_id):
            tensor = self._make_placeholder(name, False, [1], [1])
            shadow = self._make_placeholder(f"shadow_{name}", False, [1], [1])

            comp = Node()
            comp.node_type = Node.NodeType.COMP_NODE
            comp.name = f"{name}_COMP"
            comp.id = start_id
            comp.data_deps = []
            comp.ctrl_deps = []
            comp.num_ops = 1
            comp.tensor_size = 1
            comp.y_tensor_size = 1
            comp.op_type = "comp"

            send = Node()
            send.node_type = Node.NodeType.COMM_SEND_NODE
            send.name = f"{name}_Y_SEND"
            send.id = start_id + 1
            send.data_deps = [comp.id]
            send.ctrl_deps = []
            send.comm_size = 1
            send.comm_tag = tag
            send.comm_dst = send_dst
            send.y_tensor_size = 0

            recv = Node()
            recv.node_type = Node.NodeType.COMM_RECV_NODE
            recv.name = f"shadow_{name}_Y_RECV"
            recv.id = start_id + 2
            recv.data_deps = []
            recv.ctrl_deps = []
            recv.comm_size = 1
            recv.comm_tag = tag
            recv.comm_src = recv_src
            recv.y_tensor_size = 1
            return tensor, shadow, comp, send, recv

        tensor, shadow, comp, send, recv = make_pair("t", base_tag, 1)
        tensors = [tensor, shadow]
        tensor_map_nodes = {
            tensor: {
                HybridGraph.NodeType.COMP: comp,
                f"{HybridGraph.NodeType.Y_SEND}{send.comm_tag}": send,
            },
            shadow: {
                HybridGraph.NodeType.Y_RECV: recv,
            },
        }

        if extra_tag is not None:
            extra_tensor, extra_shadow, extra_comp, extra_send, extra_recv = make_pair(
                "extra_t", extra_tag, 4
            )
            tensors.extend([extra_tensor, extra_shadow])
            tensor_map_nodes[extra_tensor] = {
                HybridGraph.NodeType.COMP: extra_comp,
                f"{HybridGraph.NodeType.Y_SEND}{extra_send.comm_tag}": extra_send,
            }
            tensor_map_nodes[extra_shadow] = {
                HybridGraph.NodeType.Y_RECV: extra_recv,
            }

        graph = HybridGraph(tensors, tensor_map_nodes, {})
        return graph

    def test_microbatch_replicator_merges_before_weight_sync(self):
        from symbolic_tensor_graph.graph.grad_updater import MicroBatchReplicator

        graph, symbol_map, _ = self._make_weight_sync_graph()

        merged_graph = MicroBatchReplicator.apply(graph, symbol_map)
        tensor_map = merged_graph.get_tensor_id_map_tensor()

        self.assertNotIn("mb0.sharded_grad@0", tensor_map)
        self.assertIn("assembled_grad@0", tensor_map)

        merged_grad = tensor_map["assembled_grad@0"]
        self.assertEqual("mb0.assembled_grad@0", merged_grad.x1.id)
        self.assertEqual(
            ["mb1.assembled_grad@0", "mb2.assembled_grad@0", "mb3.assembled_grad@0"],
            [tensor.id for tensor in merged_grad.get_extra_data_dependancy()],
        )
        self.assertEqual("assembled_grad@0", tensor_map["w@0"]._grad.id)

    def test_grad_update_emits_one_collective_per_step(self):
        from symbolic_tensor_graph.graph.grad_updater import (
            GradUpdater,
            MicroBatchReplicator,
        )
        from symbolic_tensor_graph.graph.convert_chakra import ConvertChakra
        from symbolic_tensor_graph.chakra.node import Node

        graph, symbol_map, dp = self._make_weight_sync_graph()

        merged_graph = MicroBatchReplicator.apply(graph, symbol_map)
        updated_graph = GradUpdater.apply(merged_graph, inplace=True)
        updated_graph.comm_groups = {dp: [0]}

        chakra_graph = ConvertChakra.apply(updated_graph, symbol_map, [dp])
        coll_nodes = [
            node
            for node in chakra_graph.get_nodes()
            if node.node_type == Node.NodeType.COLL_COMM_NODE
        ]

        self.assertEqual(1, len(coll_nodes))
        self.assertEqual("w@1_X2_COMM", coll_nodes[0].name)
        self.assertEqual(Node.CollectiveType.ALL_REDUCE, coll_nodes[0].comm_type)

    def test_local_sgd_iteration_postprocess_reduces_dp_sync_frequency(self):
        from symbolic_tensor_graph.graph.grad_updater import (
            GradUpdater,
            LocalSGDIterationPostProcess,
            MicroBatchReplicator,
        )
        from symbolic_tensor_graph.graph.convert_chakra import ConvertChakra
        from symbolic_tensor_graph.chakra.node import Node

        graph, symbol_map, dp = self._make_weight_sync_graph()

        merged_graph = MicroBatchReplicator.apply(graph, symbol_map)
        updated_graph = GradUpdater.apply(merged_graph, inplace=True)
        updated_graph.comm_groups = {dp: [0]}

        chakra_graph = ConvertChakra.apply(updated_graph, symbol_map, [dp])
        chakra_graph = LocalSGDIterationPostProcess.replicate_iterations(
            chakra_graph,
            num_iterations=3,
            sync_interval=2,
            dp_symbol=dp,
        )

        coll_nodes = [
            node
            for node in chakra_graph.get_nodes()
            if node.node_type == Node.NodeType.COLL_COMM_NODE
        ]

        self.assertEqual(1, len(coll_nodes))
        self.assertEqual("iter1.w@1_X2_COMM", coll_nodes[0].name)
        self.assertEqual(Node.CollectiveType.ALL_REDUCE, coll_nodes[0].comm_type)

    def test_local_sgd_iteration_postprocess_rewires_non_sync_step(self):
        from symbolic_tensor_graph.graph.grad_updater import (
            GradUpdater,
            LocalSGDIterationPostProcess,
            MicroBatchReplicator,
        )
        from symbolic_tensor_graph.graph.convert_chakra import ConvertChakra

        graph, symbol_map, dp = self._make_weight_sync_graph()

        merged_graph = MicroBatchReplicator.apply(graph, symbol_map)
        updated_graph = GradUpdater.apply(merged_graph, inplace=True)
        updated_graph.comm_groups = {dp: [0]}

        chakra_graph = ConvertChakra.apply(updated_graph, symbol_map, [dp])
        chakra_graph = LocalSGDIterationPostProcess.replicate_iterations(
            chakra_graph,
            num_iterations=2,
            sync_interval=2,
            dp_symbol=dp,
        )

        node_map = {node.name: node for node in chakra_graph.get_nodes()}

        self.assertNotIn("w@1_X2_COMM", node_map)
        self.assertIn("assembled_grad@0_COMP", node_map)
        self.assertIn("w@1_COMP", node_map)
        self.assertEqual(
            [node_map["assembled_grad@0_COMP"].id],
            node_map["w@1_COMP"].data_deps,
        )

        self.assertIn("iter0_to_iter1_BARRIER", node_map)
        barrier = node_map["iter0_to_iter1_BARRIER"]
        self.assertEqual([node_map["w@1_COMP"].id], barrier.data_deps)
        self.assertGreater(barrier.num_ops, 0)
        self.assertGreater(barrier.tensor_size, 0)
        self.assertGreater(barrier.y_tensor_size, 0)
        self.assertIn("iter1.assembled_grad@0_COMP", node_map)
        self.assertIn(barrier.id, node_map["iter1.assembled_grad@0_COMP"].data_deps)

    def test_local_sgd_iteration_postprocess_offsets_point_to_point_tags(self):
        from symbolic_tensor_graph.graph.grad_updater import LocalSGDIterationPostProcess

        graph = self._make_point_to_point_graph()
        graph = LocalSGDIterationPostProcess.replicate_iterations(
            graph,
            num_iterations=2,
            sync_interval=1,
            dp_symbol=sp.symbols("dp"),
        )

        node_map = {node.name: node for node in graph.get_nodes()}

        self.assertEqual(7, node_map["t_Y_SEND"].comm_tag)
        self.assertEqual(7, node_map["shadow_t_Y_RECV"].comm_tag)
        self.assertEqual(8, node_map["iter1.t_Y_SEND"].comm_tag)
        self.assertEqual(8, node_map["iter1.shadow_t_Y_RECV"].comm_tag)

    def test_local_sgd_iteration_postprocess_uses_global_stride_for_bundled_graphs(self):
        from symbolic_tensor_graph.graph.grad_updater import LocalSGDIterationPostProcess
        from symbolic_tensor_graph.graph.graph import BundledHybridGraph

        rank0 = (("rank", 0),)
        rank1 = (("rank", 1),)
        bundled_graph = BundledHybridGraph(
            {
                rank0: self._make_point_to_point_graph(
                    send_dst=1, recv_src=1, base_tag=7, extra_tag=30
                ),
                rank1: self._make_point_to_point_graph(
                    send_dst=0, recv_src=0, base_tag=7, extra_tag=90
                ),
            },
            remote_parent_shadow_pairs=[],
            spatial_parallel_dims=[],
            temporal_parallel_dims=[],
            symbol_value_dict={},
            readable_rank_map_number_rank={rank0: 0, rank1: 1},
        )

        bundled_graph = LocalSGDIterationPostProcess.apply(
            bundled_graph,
            num_iterations=2,
            sync_interval=1,
            dp_symbol=sp.symbols("dp"),
        )

        rank0_nodes = {node.name: node for node in bundled_graph.graphs[rank0].get_nodes()}
        rank1_nodes = {node.name: node for node in bundled_graph.graphs[rank1].get_nodes()}

        self.assertEqual(7, rank0_nodes["t_Y_SEND"].comm_tag)
        self.assertEqual(7, rank1_nodes["shadow_t_Y_RECV"].comm_tag)
        self.assertEqual(91, rank0_nodes["iter1.t_Y_SEND"].comm_tag)
        self.assertEqual(91, rank1_nodes["iter1.shadow_t_Y_RECV"].comm_tag)
        self.assertEqual(91, rank1_nodes["iter1.t_Y_SEND"].comm_tag)
        self.assertEqual(91, rank0_nodes["iter1.shadow_t_Y_RECV"].comm_tag)
