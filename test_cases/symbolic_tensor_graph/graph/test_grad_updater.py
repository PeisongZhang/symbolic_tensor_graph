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
