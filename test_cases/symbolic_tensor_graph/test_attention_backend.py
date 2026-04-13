import unittest


class TestAttentionBackend(unittest.TestCase):
    def test_llama_attention_backend_resolution(self):
        from models.stage1.llama_model import _resolve_gqa_kernel_path

        backend, path = _resolve_gqa_kernel_path(attention_backend="standard")
        self.assertEqual("standard", backend)
        self.assertTrue(path.endswith("group_query_attention_kernel.csv"))

        backend, path = _resolve_gqa_kernel_path(flash_attention=True)
        self.assertEqual("flash", backend)
        self.assertTrue(path.endswith("group_query_attention_kernel_flash.csv"))

        backend, path = _resolve_gqa_kernel_path()
        self.assertEqual("fused", backend)
        self.assertTrue(path.endswith("group_query_attention_kernel_fused.csv"))

    def test_gpt_attention_backend_resolution(self):
        from models.stage1.gpt_model import _resolve_gqa_kernel_path

        backend, path = _resolve_gqa_kernel_path(
            tpsp=False, attention_backend="standard"
        )
        self.assertEqual("standard", backend)
        self.assertTrue(path.endswith("tp_gpt/group_query_attention_kernel.csv"))

        backend, path = _resolve_gqa_kernel_path(tpsp=True, flash_attention=True)
        self.assertEqual("flash", backend)
        self.assertTrue(path.endswith("tpsp_gpt/group_query_attention_kernel_flash.csv"))

    def test_invalid_attention_backend_is_rejected(self):
        from models.stage1.llama_model import _resolve_gqa_kernel_path

        with self.assertRaises(ValueError):
            _resolve_gqa_kernel_path(attention_backend="invalid")
