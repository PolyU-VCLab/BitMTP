"""D4 public schedule and strict policy checks; no CUDA allocation."""
from types import SimpleNamespace
import unittest

from bonsai_sft.gpu.contracts import require_kernel_shape
from bonsai_sft.gpu.run_generation import make_parser, validate_options


class BonsaiD4Contracts(unittest.TestCase):
    def args(self, mode, precision='native'):
        return make_parser().parse_args([
            '--ckpt', '/unused/model', '--hf-ckpt', '/unused/tokenizer',
            '--block-size', '4', '--mode', mode, '--prompt', 'hello',
            '--draft-precision', precision,
        ])

    def test_d4_native_spec_and_fp8_hybrid_schedule(self):
        for mode, precision, expected_roles in (
                ('ar', 'native', ('ar',)),
                ('conf', 'fp8', ('ar', 'draft_fp8')),
                ('spec', 'native', ('ar', 'draft', 'verify')),
                ('hybrid', 'fp8', ('ar', 'draft_fp8', 'verify'))):
            args = self.args(mode, precision)
            widths, roles, chain = validate_options(args)
            self.assertEqual(widths, (1,) if mode == 'ar' else (1,4,5,6,7,8,9))
            self.assertEqual(roles, expected_roles)
            self.assertEqual(chain, 3)
            self.assertEqual(args.margin, 0)
            for m in widths:
                require_kernel_shape('t2g128', m, 151936, 2048)

    def test_d4_does_not_imply_4b_native_width_support(self):
        for m in (4,5,6,7):
            with self.assertRaisesRegex(RuntimeError, 'unsupported t2g128'):
                require_kernel_shape('t2g128', m, 151936, 2560)

    def test_d4_pending_widths_preserve_anchor_capacity(self):
        from bonsai_sft.gpu.fast_generate import _select_M_for_pending
        atlas = SimpleNamespace(D=4, m_set=(1,4,5,6,7,8,9))
        for pending, expected in ((1,5),(2,6),(3,7),(4,8),(5,9)):
            self.assertEqual(_select_M_for_pending(atlas, pending), expected)

    def test_d4_rejects_nonzero_verification_margin(self):
        args = self.args('spec')
        for margin in (0.5, -0.5, 1e-20, float('nan'), float('inf')):
            args.margin = margin
            with self.assertRaises(ValueError):
                validate_options(args)


if __name__ == '__main__':
    unittest.main()
