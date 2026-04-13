import unittest
import sympy as sp


class TestTensor(unittest.TestCase):
    def _assert_expr_equal(self, lhs, rhs):
        self.assertEqual(0, sp.simplify(lhs - rhs))

    def _assert_expr_list_equal(self, lhs, rhs):
        self.assertEqual(len(lhs), len(rhs))
        for lhs_item, rhs_item in zip(lhs, rhs):
            self._assert_expr_equal(lhs_item, rhs_item)

    def test_parse_to_records1(self):
        validation_file = "./sharding_spreadsheets/module/linear.csv"
        resave_1_file = "/tmp/TestTensor_test_parse_to_records1_resave1.csv"
        resave_2_file = "/tmp/TestTensor_test_parse_to_records1_resave2.csv"
        from symbolic_tensor_graph.tensor import Tensor

        tensors1 = Tensor.parse_records(validation_file)
        Tensor.to_records(tensors1, resave_1_file)
        tensors2 = Tensor.parse_records(resave_1_file)
        for tensor1, tensor2 in zip(tensors1, tensors2):
            self.assertEqual(tensor1._to_record(), tensor2._to_record())
        Tensor.to_records(tensors2, resave_2_file)

    def test_op_handler1(self):
        validation_file = "./sharding_spreadsheets/module/linear.csv"
        from symbolic_tensor_graph.tensor import Tensor
        import sympy as sp

        B, M, NN = sp.symbols("Batch InFeat OutFeat")
        ground_truth = {
            "x": ([B, M], [1], 0),
            "w": ([M, NN], [1], 0),
            "y": ([B, NN], [M], 2 * B * M * NN),
            "dy": ([B, NN], [1], 0),
            "dw": ([M, NN], [B], 2 * B * M * NN),
            "dx": ([B, M], [NN], 2 * B * M * NN),
        }
        tensors = Tensor.parse_records(validation_file)
        for tensor in tensors:
            gt_y_shape, gt_y_hidden, gt_ops = ground_truth[tensor.name]
            self.assertEqual(tensor.y_shape, gt_y_shape)
            self._assert_expr_list_equal(tensor.y_hidden, gt_y_hidden)
            self._assert_expr_equal(tensor.ops, gt_ops)

    def test_parse_to_records2(self):
        validation_file = "./sharding_spreadsheets/test/mlp2.csv"
        resave_1_file = "/tmp/TestTensor_test_parse_to_records2_resave1.csv"
        resave_2_file = "/tmp/TestTensor_test_parse_to_records2_resave2.csv"
        from symbolic_tensor_graph.tensor import Tensor

        tensors1 = Tensor.parse_records(validation_file)
        Tensor.to_records(tensors1, resave_1_file)
        tensors2 = Tensor.parse_records(resave_1_file)
        for tensor1, tensor2 in zip(tensors1, tensors2):
            self.assertEqual(tensor1._to_record(), tensor2._to_record())
        Tensor.to_records(tensors2, resave_2_file)

    def test_op_handler2(self):
        validation_file = "./sharding_spreadsheets/test/mlp2.csv"
        from symbolic_tensor_graph.tensor import Tensor
        import sympy as sp

        B, N0, N1, N2 = sp.symbols("Batch Feat0 Feat1 Feat2")
        ground_truth = {
            "x0": ([B, N0], [1], 0),
            "w1": ([N0, N1], [1], 0),
            "x1": ([B, N1], [N0], 2 * B * N1 * N0),
            "w2": ([N1, N2], [1], 0),
            "x2": ([B, N2], [N1], 2 * B * N2 * N1),
            "dx0": ([B, N0], [N1], 2 * B * N0 * N1),
            "dw1": ([N0, N1], [B], 2 * B * N0 * N1),
            "dx1": ([B, N1], [N2], 2 * B * N2 * N1),
            "dw2": ([N1, N2], [B], 2 * B * N2 * N1),
            "dx2": ([B, N2], [1], 0),
        }

        tensors = Tensor.parse_records(validation_file)
        for tensor in tensors:
            gt_y_shape, gt_y_hidden, gt_ops = ground_truth[tensor.name]
            self.assertEqual(tensor.y_shape, gt_y_shape)
            self._assert_expr_list_equal(tensor.y_hidden, gt_y_hidden)
            self._assert_expr_equal(tensor.ops, gt_ops)

    def test_op_handler3(self):
        validation_file = "./sharding_spreadsheets/test/test_ops.csv"
        from symbolic_tensor_graph.tensor import Tensor
        import sympy as sp

        B, S, M = sp.symbols("Batch Seq Model")
        ground_truth = {
            "x": ([B, S, M], [1], 0),
            "w": ([M, M], [1], 0),
            "y": ([B, S, M], [M], 2 * B * S * M * M),
            "res": ([B, S, M], [1], B * S * M),
            "norm": ([B, S, M], [1], 5.0 * B * S * M),
            "reshape": ([M * M], [1], M * M),
            "x2": ([B, S, M], [1], 0),
        }
        tensors = Tensor.parse_records(validation_file)
        for tensor in tensors:
            gt_y_shape, gt_y_hidden, gt_ops = ground_truth[tensor.name]
            self.assertEqual(tensor.y_shape, gt_y_shape)
            self._assert_expr_list_equal(tensor.y_hidden, gt_y_hidden)
            self._assert_expr_equal(tensor.ops, gt_ops)

    def test_visualize1(self):
        validation_file = "./sharding_spreadsheets/module/linear.csv"
        visualize_file = "/tmp/TestTensor_test_visualize1.pdf"
        from symbolic_tensor_graph.tensor import Tensor

        tensors = Tensor.parse_records(validation_file)
        Tensor.visualize(tensors, visualize_file)

    def test_visualize2(self):
        validation_file = "./sharding_spreadsheets/test/mlp2.csv"
        visualize_file = "/tmp/TestTensor_test_visualize2.pdf"
        from symbolic_tensor_graph.tensor import Tensor

        tensors = Tensor.parse_records(validation_file)
        Tensor.visualize(tensors, visualize_file)

    def test_visualize3(self):
        validation_file = "./sharding_spreadsheets/module/multi_head_attention.csv"
        visualize_file = "/tmp/TestTensor_test_visualize3.pdf"
        from symbolic_tensor_graph.tensor import Tensor

        tensors = Tensor.parse_records(validation_file)
        Tensor.visualize(tensors, visualize_file)


if __name__ == "__main__":
    unittest.main()
