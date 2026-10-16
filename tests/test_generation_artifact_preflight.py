"""CPU-only tokenizer preflight and active pending-width selection checks."""
import ast
import builtins
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bonsai_sft.gpu import run_generation


class TokenizerPreflight(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.checkpoint = root / 'checkpoint'
        self.tokenizer = root / 'tokenizer'
        self.checkpoint.mkdir()
        self.tokenizer.mkdir()
        config = dict(dim=2048, n_layers=28, n_heads=16, n_kv_heads=8,
                      head_dim=128, ffn_dim=6144, vocab_size=151669,
                      tie_word_embeddings=True, block_size=8,
                      kernel_kind='t2g128', quantize_embedding=True)
        (self.checkpoint / 'bonsai_gpu_config.json').write_text(json.dumps(config))
        (self.checkpoint / 'model_state_t2g128.pt').write_bytes(b'preflight fixture')
        (self.tokenizer / 'tokenizer_config.json').write_text('{}')
        module_root = root / 'module'
        library = module_root / 't2g128_kernels/libbonsai_t2.so'
        library.parent.mkdir(parents=True)
        library.write_bytes(b'preflight fixture; never loaded')
        module_path = patch.object(run_generation, '__file__', str(module_root / 'run_generation.py'))
        module_path.start()
        self.addCleanup(module_path.stop)
        self.argv = ['--ckpt', str(self.checkpoint), '--hf-ckpt', str(self.tokenizer),
                     '--block-size', '8', '--mode', 'ar', '--prompt', 'hello', '--check-only']

    def add_files(self, *names):
        for name in names:
            (self.tokenizer / name).write_text('{}')

    def checked(self):
        original_import = builtins.__import__
        def no_torch(name, *args, **kwargs):
            if name == 'torch' or name.startswith('torch.'):
                raise AssertionError('check-only must not import torch')
            return original_import(name, *args, **kwargs)
        output = io.StringIO()
        with patch('builtins.__import__', side_effect=no_torch), contextlib.redirect_stdout(output):
            run_generation.main(self.argv)
        result = json.loads(output.getvalue())
        self.assertEqual(result['status'], 'local_artifacts_and_options_checked')
        self.assertFalse(result['gpu_executed'])

    def rejected(self):
        args = run_generation.make_parser().parse_args(self.argv)
        with self.assertRaisesRegex(ValueError, 'tokenizer.json or both vocab.json and merges.txt'):
            run_generation.validate_artifacts(args, (1,))

    def test_tokenizer_json_passes_without_bpe_sidecars_or_torch(self):
        self.add_files('tokenizer.json')
        self.checked()

    def test_vocab_and_merges_pass_without_tokenizer_json_or_torch(self):
        self.add_files('vocab.json', 'merges.txt')
        self.checked()

    def test_vocab_alone_is_rejected(self):
        self.add_files('vocab.json')
        self.rejected()

    def test_merges_alone_is_rejected(self):
        self.add_files('merges.txt')
        self.rejected()

    def test_sentencepiece_file_alone_is_rejected(self):
        self.add_files('tokenizer.model')
        self.rejected()

    def test_sentencepiece_file_does_not_complete_vocab(self):
        self.add_files('tokenizer.model', 'vocab.json')
        self.rejected()

    def test_no_tokenizer_representation_is_rejected(self):
        self.rejected()

    def test_check_only_cli_reports_incomplete_bpe(self):
        self.add_files('vocab.json')
        error = io.StringIO()
        with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as raised:
            run_generation.main(self.argv)
        self.assertEqual(raised.exception.code, 2)
        self.assertIn('both vocab.json and merges.txt', error.getvalue())


class PendingWidthSelection(unittest.TestCase):
    @staticmethod
    def gpu_root():
        return Path(__file__).resolve().parents[1] / 'src/bonsai_sft/gpu'

    def test_obsolete_atlas_selector_is_removed(self):
        tree = ast.parse((self.gpu_root() / 'fast_runtime.py').read_text())
        atlas = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'GraphAtlas')
        self.assertNotIn('select_M', [node.name for node in atlas.body if isinstance(node, ast.FunctionDef)])

    def active_selector(self):
        path = self.gpu_root() / 'fast_generate.py'
        tree = ast.parse(path.read_text())
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_select_M_for_pending')
        namespace = {'GraphAtlas': object}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        return namespace['_select_M_for_pending']

    def test_active_d4_selector_covers_all_reachable_pending_lengths(self):
        select = self.active_selector()
        atlas = SimpleNamespace(D=4, m_set=(1, 4, 5, 6, 7, 8, 9))
        self.assertEqual([select(atlas, n) for n in range(1, 6)], [5, 6, 7, 8, 9])

    def test_active_d8_selector_covers_all_reachable_pending_lengths(self):
        select = self.active_selector()
        atlas = SimpleNamespace(D=8, m_set=(1, 8, 9, 10, 12, 16, 17))
        self.assertEqual([select(atlas, n) for n in range(1, 10)], [9, 10, 12, 12, 16, 16, 16, 16, 17])


if __name__ == '__main__':
    unittest.main()
