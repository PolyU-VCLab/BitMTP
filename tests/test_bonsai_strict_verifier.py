"""Strict-verifier semantics and device-role boundaries, without GPU execution."""
import ast
import contextlib
import inspect
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'CPU torch required')
class StrictVerifierTests(unittest.TestCase):
    def test_tied_logits_do_not_admit_a_different_token_id(self):
        from bonsai_sft.gpu.fast_generate import _fused_verify_accept
        accept = getattr(_fused_verify_accept, '_torchdynamo_orig_callable', _fused_verify_accept)
        tied_logits = torch.tensor([[3.0, 3.0], [0.0, 0.0], [2.0, 2.0]])
        verifier = tied_logits.argmax(-1)
        self.assertEqual(accept(torch.tensor([1, 0]), verifier, 2).item(), 0)
        self.assertEqual(accept(torch.tensor([0, 1]), verifier, 2).item(), 1)
        self.assertEqual(accept(torch.tensor([0, 0]), verifier, 2).item(), 2)

    def test_acceptance_uses_only_ids_and_longest_prefix(self):
        from bonsai_sft.gpu.fast_generate import _fused_verify_accept
        accept = getattr(_fused_verify_accept, '_torchdynamo_orig_callable', _fused_verify_accept)
        self.assertEqual(list(inspect.signature(accept).parameters), ['draft_tokens', 'verify_tokens', 'D'])
        verify = torch.arange(9)
        for prefix in range(9):
            draft = torch.arange(8)
            if prefix < 8:
                draft[prefix] = 99
            self.assertEqual(accept(draft, verify, 8).item(), prefix)

    def test_every_nonzero_margin_is_rejected_before_gpu_use(self):
        from bonsai_sft.gpu.fast_generate import fast_speculative_generate, fast_hybrid_generate
        for generate in (fast_speculative_generate, fast_hybrid_generate):
            for margin in (-0.5, 1e-20, 0.5, float('inf'), float('nan')):
                with self.subTest(generate=generate.__name__, margin=margin):
                    with self.assertRaisesRegex(ValueError, 'requires rabv_margin=0'):
                        generate(None, None, None, rabv_margin=margin)

    def test_nonnative_verifier_flags_rejected_before_gpu_use(self):
        from bonsai_sft.gpu.fast_generate import fast_speculative_generate, fast_hybrid_generate
        for generate in (fast_speculative_generate, fast_hybrid_generate):
            for flag in ('use_fp8_verify', 'use_q1_verify', 'use_w1a16_verify'):
                with self.assertRaisesRegex(ValueError, 'native verifier projection'):
                    generate(None, None, None, **{flag: True})

    def test_no_gap_or_svr_acceptance_code_remains(self):
        from bonsai_sft.gpu import fast_generate
        source = Path(fast_generate.__file__).read_text()
        tree = ast.parse(source)
        self.assertNotIn('_fused_draft_chain_svr', source)
        self.assertNotIn('svr_gap', source)
        self.assertNotIn('verify_logit - draft_logit', source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_fused_verify_accept')
        self.assertFalse(any(isinstance(n, ast.If) for n in ast.walk(fn)))
        self.assertNotIn('svr_enabled', inspect.signature(fast_generate.fast_hybrid_generate).parameters)

    def test_eager_verifier_also_uses_ids_with_ties(self):
        from bonsai_sft.gpu.generate import _exact_m1_verify
        class FakeModel:
            def forward_sdlm_decode(self, **kwargs):
                # Different token 1 ties with selected argmax 0, but must reject.
                return torch.tensor([[1.0, 1.0]]), [torch.zeros(1, 1, 1, 1)], [torch.zeros(1, 1, 1, 1)]
        cache = [(torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1))]
        accepted, bonus, _, _ = _exact_m1_verify(FakeModel(), cache, 1,
                                                torch.tensor([0]), torch.tensor([1]))
        self.assertEqual(accepted, [])
        self.assertEqual(bonus, 0)

    def test_sm86_native_verification_and_bf16_draft_roles(self):
        from bonsai_sft.gpu.contracts import validate_graph_roles
        validate_graph_roles(('ar', 'draft_bf16', 'verify'), (8, 6))
        validate_graph_roles(('ar', 'draft', 'verify'), (8, 6))
        with self.assertRaisesRegex(ValueError, 'FP8 draft'):
            validate_graph_roles(('ar', 'draft_fp8', 'verify'), (8, 6))
        validate_graph_roles(('ar', 'draft_fp8', 'verify'), (9, 0))
        for role in ('verify_fp8', 'verify_q1', 'verify_w1a16'):
            with self.assertRaisesRegex(ValueError, 'unsupported strict graph roles'):
                validate_graph_roles(('ar', role), (9, 0))

    def test_bf16_draft_available_for_all_mtp_modes(self):
        from bonsai_sft.gpu import fast_generate
        from bonsai_sft.gpu.run_generation import make_parser, validate_options
        for mode, generate in (('conf', fast_generate.fast_confidence_generate),
                               ('spec', fast_generate.fast_speculative_generate),
                               ('hybrid', fast_generate.fast_hybrid_generate)):
            self.assertIn('use_bf16_draft', inspect.signature(generate).parameters)
            args = make_parser().parse_args(['--ckpt', '/unused/model', '--hf-ckpt', '/unused/tokenizer',
                                             '--block-size', '8', '--mode', mode, '--prompt', 'hello',
                                             '--draft-precision', 'bf16', '--verify-precision', 'native'])
            self.assertIn('draft_bf16', validate_options(args)[1])

    def test_cli_has_no_nonnative_verifier_selection(self):
        from bonsai_sft.gpu.run_generation import make_parser
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            make_parser().parse_args(['--ckpt', '/unused/model', '--hf-ckpt', '/unused/tokenizer',
                                      '--block-size', '8', '--mode', 'spec', '--prompt', 'hello',
                                      '--verify-precision', 'fp8'])


if __name__ == '__main__':
    unittest.main()
