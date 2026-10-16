"""CPU-only checks for the explicit Orin runtime, without loading weights."""
import builtins
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from bonsai_sft.gpu import runtime_config as runtime
from bonsai_sft.gpu import run_generation as cli


class OrinProfiles(unittest.TestCase):
    def args(self, profile='orin-d8', mode='hybrid', block=8, extra=()):
        return cli.make_parser().parse_args([
            '--ckpt', '/unused/model', '--hf-ckpt', '/unused/hf',
            '--block-size', str(block), '--mode', mode, '--prompt', 'hello',
            '--runtime-profile', profile, *extra])

    def test_three_consumed_profiles(self):
        for profile, mode, block, widths in (
                ('orin-sft', 'ar', 8, (1,)),
                ('orin-d4', 'hybrid', 4, (1, 4, 5, 6, 7, 8, 9)),
                ('orin-d8', 'hybrid', 8, (1, 8, 9, 10, 12, 16, 17))):
            args = self.args(profile, mode, block)
            actual, roles, gate = cli.validate_options(args)
            self.assertEqual(actual, widths)
            self.assertEqual(gate, 3)
            self.assertEqual(roles, ('ar',) if mode == 'ar' else ('ar', 'draft', 'verify'))

    def test_standard_defaults_are_unchanged(self):
        self.assertEqual(cli.validate_options(self.args('standard'))[2], 6)
        for precision, role in (('fp8', 'draft_fp8'), ('bf16', 'draft_bf16')):
            args = self.args('standard', extra=('--draft-precision', precision))
            self.assertIn(role, cli.validate_options(args)[1])

    def test_orin_mtp_modes_use_native_roles(self):
        for profile, block in (('orin-d4', 4), ('orin-d8', 8)):
            for mode in ('conf', 'spec', 'hybrid'):
                roles = cli.validate_options(self.args(profile, mode, block))[1]
                self.assertIn('draft', roles)
                self.assertEqual('verify' in roles, mode != 'conf')

    def test_profile_conflicts_reject(self):
        for args in (self.args('orin-d4'), self.args('orin-sft'),
                     self.args(mode='ar'),
                     self.args(extra=('--min-chain', '6')),
                     self.args(extra=('--max-context', '2048')),
                     self.args(extra=('--threshold', '.8')),
                     self.args(extra=('--draft-precision', 'fp8')),
                     self.args(extra=('--draft-precision', 'bf16')),
                     self.args(extra=('--margin', '.0001'))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                cli.validate_options(args)

    def test_geometry_and_device_reject_unsupported_scope(self):
        for config in ({'dim': 2560, 'kernel_kind': 't2g128'},
                       {'dim': 2048, 'kernel_kind': 'q1'}):
            with self.assertRaises(ValueError):
                runtime.validate_profile_model('orin-d8', config)
        runtime.validate_profile_model('orin-d8', {'dim': 2048, 'kernel_kind': 't2g128'})
        runtime.validate_profile_device('orin-d8', (8, 7))
        for sm in ((8, 6), (9, 0)):
            with self.assertRaises(ValueError):
                runtime.validate_profile_device('orin-d8', sm)
            runtime.validate_profile_device('standard', sm)

    def test_check_only_does_not_import_torch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ckpt, hf = root / 'ckpt', root / 'hf'
            ckpt.mkdir()
            hf.mkdir()
            config = dict(dim=2048, n_layers=28, n_heads=16, n_kv_heads=8,
                          head_dim=128, ffn_dim=6144, vocab_size=151669,
                          tie_word_embeddings=True, block_size=8, kernel_kind='t2g128')
            (ckpt / 'bonsai_gpu_config.json').write_text(json.dumps(config))
            (ckpt / 'model_state_t2g128.pt').write_bytes(b'never loaded')
            (hf / 'tokenizer.json').write_text('{}')
            (hf / 'tokenizer_config.json').write_text('{}')
            (root / 't2g128_kernels').mkdir()
            (root / 't2g128_kernels/libbonsai_t2.so').write_bytes(b'never loaded')
            original = builtins.__import__
            def checked_import(name, *args, **kwargs):
                if name == 'torch' or name.startswith('torch.'):
                    raise AssertionError('check-only imported torch')
                return original(name, *args, **kwargs)
            stream = io.StringIO()
            with patch.object(cli, '__file__', str(root / 'run_generation.py')), \
                    patch('builtins.__import__', side_effect=checked_import), \
                    contextlib.redirect_stdout(stream):
                cli.main(['--ckpt', str(ckpt), '--hf-ckpt', str(hf), '--block-size', '8',
                          '--mode', 'hybrid', '--runtime-profile', 'orin-d8',
                          '--prompt', 'hello', '--check-only'])
            result = json.loads(stream.getvalue())
            self.assertFalse(result['gpu_executed'])
            self.assertEqual(result['runtime_profile'], 'orin-d8')
            self.assertEqual(result['min_chain'], 3)


class CompilePolicy(unittest.TestCase):
    def test_standard_delegates_without_changing_kwargs(self):
        import torch
        def example(value):
            return value
        original = torch.compile
        with patch.object(runtime, '_profile', 'standard'), patch.object(torch, 'compile') as compile:
            runtime.compile_helper(example)
            compile.assert_called_once_with(example)
            compile.reset_mock()
            runtime.compile_helper(mode='max-autotune-no-cudagraphs', dynamic=False)
            compile.assert_called_once_with(mode='max-autotune-no-cudagraphs', dynamic=False)
        self.assertIs(torch.compile, original)

    def test_orin_helper_identity_never_calls_compile(self):
        import torch
        def example(value):
            return value
        with patch.object(runtime, '_profile', 'orin-d8'), patch.object(torch, 'compile') as compile:
            self.assertIs(runtime.compile_helper(example), example)
            self.assertIs(runtime.compile_helper(mode='reduce-overhead')(example), example)
            compile.assert_not_called()

    def test_orin_imports_and_eager_predicates_in_clean_process(self):
        code = '''
import builtins
import torch
from bonsai_sft.gpu.runtime_config import select_runtime_profile
select_runtime_profile('orin-d8')
original_import = builtins.__import__
original_compile = torch.compile
def reject_optional(name, *args, **kwargs):
    if name.startswith(('triton', 'xformers', 'torch._dynamo', 'torch.nn.attention.flex_attention')):
        raise AssertionError('forbidden Orin import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = reject_optional
from bonsai_sft.gpu import model, fast_runtime, fast_generate
assert torch.compile is original_compile
assert not model._HAS_FLEX and not model._HAS_XFORMERS and not fast_runtime._FLEX_AVAILABLE
assert not torch.cuda.is_initialized()
verify = torch.arange(9)
for prefix in range(9):
    draft = torch.arange(8)
    if prefix < 8:
        draft[prefix] = 99
    assert fast_generate._fused_verify_accept(draft, verify, 8).item() == prefix
logits = torch.tensor([[0., 0.], [20., 0.], [0., 0.], [20., 0.]])
tokens, length = fast_generate._fused_draft_chain(logits, .9)
assert tokens.tolist() == [0, 0, 0, 0] and length.item() == 2
try:
    select_runtime_profile('standard')
except RuntimeError:
    pass
else:
    raise AssertionError('late profile change accepted')
assert not torch.cuda.is_initialized()
'''
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        result = subprocess.run([sys.executable, '-B', '-c', code], env=env,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
