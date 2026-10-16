"""Focused tensor/role tests, always on CPU or the allocation-free meta device."""
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'CPU torch is required')
class BonsaiCPUContracts(unittest.TestCase):
    def test_verifier_anchor_is_not_counted_twice(self):
        from bonsai_sft.gpu.fast_runtime import GraphAtlas
        atlas = GraphAtlas.__new__(GraphAtlas)
        sigma, m, cache_len = 12, 9, 5
        template = torch.full((1, 1, m, sigma + m), float('-inf'))
        for row in range(m):
            template[:, :, row, sigma:sigma + row + 1] = 0
        atlas.verify_templates = {(m, sigma): template}
        for role in ('verify',):
            graph = SimpleNamespace(M=m, sigma=sigma, role=role,
                                    attn_mask=torch.zeros_like(template))
            atlas._prepare_verify(graph, cache_len)
            self.assertTrue(torch.isneginf(graph.attn_mask[..., cache_len - 1]).all())
            self.assertTrue((graph.attn_mask[..., :cache_len - 1] == 0).all())
            self.assertTrue((graph.attn_mask[..., sigma] == 0).all())
            self.assertTrue(torch.isneginf(graph.attn_mask[..., 0, sigma + 1:]).all())
            with self.assertRaisesRegex(RuntimeError, 'committed anchor'):
                atlas._prepare_verify(graph, 0)

    def test_ar_keeps_last_history_slot(self):
        from bonsai_sft.gpu.fast_runtime import GraphAtlas
        atlas = GraphAtlas.__new__(GraphAtlas)
        sigma, cache_len = 12, 5
        template = torch.full((1, 1, 1, sigma + 1), float('-inf'))
        template[..., sigma] = 0
        atlas.verify_templates = {(1, sigma): template}
        graph = SimpleNamespace(M=1, sigma=sigma, role='ar',
                                attn_mask=torch.zeros_like(template))
        atlas._prepare_verify(graph, cache_len)
        self.assertEqual(graph.attn_mask[..., cache_len - 1].item(), 0)

    def test_public_4b_guard_precedes_cuda_or_library_load(self):
        from bonsai_sft.gpu import model
        for m in (4, 5, 6, 7):
            # Meta tensors allocate no host/GPU storage, even for the large head.
            a = torch.empty((m, 2560), dtype=torch.int8, device='meta')
            b = torch.empty((151936, 640), dtype=torch.int8, device='meta')
            s = torch.empty((m, 1), dtype=torch.float32, device='meta')
            ws = torch.empty((151936, 20), dtype=torch.bfloat16, device='meta')
            with self.assertRaisesRegex(RuntimeError, 'unsupported t2g128 kernel shape'):
                model.t2g128_int8xint2_linear(a, b, s, ws)
        self.assertIsNone(model._t2_lib)

    def test_explicit_draft_and_verifier_roles(self):
        from bonsai_sft.gpu.fast_generate import _draft_role, _verify_role
        atlas = SimpleNamespace(roles=('draft', 'draft_fp8', 'draft_bf16',
                                       'verify', 'verify_q1', 'verify_fp8', 'verify_w1a16'))
        self.assertEqual(_draft_role(atlas, False), 'draft')
        self.assertEqual(_draft_role(atlas, False, True), 'draft_bf16')
        self.assertEqual(_draft_role(atlas, True), 'draft_fp8')
        self.assertEqual(_verify_role(atlas, False), 'verify')
        with self.assertRaisesRegex(ValueError, 'native verifier'):
            _verify_role(atlas, False, True)
        with self.assertRaisesRegex(RuntimeError, 'mutually exclusive'):
            _draft_role(atlas, True, True)
        with self.assertRaisesRegex(ValueError, 'native verifier'):
            _verify_role(atlas, True, True)

    def cli_args(self, *extra):
        from bonsai_sft.gpu.run_generation import make_parser
        return make_parser().parse_args(['--ckpt', '/unused/model', '--hf-ckpt', '/unused/tokenizer',
                                         '--block-size', '8', '--mode', 'hybrid', '--prompt', 'hello', *extra])

    def test_cli_explicit_defaults(self):
        from bonsai_sft.gpu.run_generation import validate_options
        args = self.cli_args()
        widths, roles, min_chain = validate_options(args)
        self.assertEqual(args.margin, 0.0)
        self.assertEqual(min_chain, 6)
        self.assertEqual(roles, ('ar', 'draft', 'verify'))
        self.assertEqual(widths, (1, 8, 9, 10, 12, 16, 17))
        args.block_size = 4
        self.assertEqual(validate_options(args)[2], 3)

    def test_cli_rejects_ignored_options(self):
        from bonsai_sft.gpu.run_generation import validate_options
        for extra in (('--mode', 'ar', '--margin', '0.5'),
                      ('--mode', 'conf', '--min-chain', '3')):
            with self.assertRaises(ValueError):
                validate_options(self.cli_args(*extra))

    def test_cli_local_artifact_check_fails_before_cuda(self):
        from bonsai_sft.gpu.run_generation import validate_artifacts
        with self.assertRaisesRegex(ValueError, 'missing GPU checkpoint configuration'):
            validate_artifacts(self.cli_args(), (1, 8, 9))


if __name__ == '__main__':
    unittest.main()
