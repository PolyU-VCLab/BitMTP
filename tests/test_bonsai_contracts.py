"""Pure CPU/source contract checks; no torch installation or GPU required."""
import ast
import importlib.util
import re
from pathlib import Path
import unittest


PACKAGE = Path(__file__).resolve().parents[1] / 'src' / 'bonsai_sft'
spec = importlib.util.spec_from_file_location('bonsai_contracts', PACKAGE / 'gpu/contracts.py')
contracts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contracts)


class ShapeContracts(unittest.TestCase):
    def test_17b_registered_shapes(self):
        for kind in ('q1', 't2g128'):
            for m in contracts.Q1_M:
                for n, k in contracts.BONSAI_17B_NK:
                    self.assertTrue(contracts.supports_kernel_shape(kind, m, n, k))

    def test_4b_shape_guard_is_joint(self):
        for n, k in contracts.T2_4B_NK:
            for m in contracts.T2_4B_M:
                self.assertTrue(contracts.supports_kernel_shape('t2g128', m, n, k))
            for m in (0, 2, 3, 4, 5, 6, 7, 11, 13, 18):
                with self.assertRaisesRegex(RuntimeError, 'unsupported t2g128'):
                    contracts.require_kernel_shape('t2g128', m, n, k)
            self.assertFalse(contracts.supports_kernel_shape('q1', 1, n, k))

    def test_4b_unmerged_fallback_shapes(self):
        for n, k in ((4096, 2560), (9728, 2560)):
            for m in contracts.Q1_M:
                self.assertFalse(contracts.supports_kernel_shape('t2g128', m, n, k))

    def test_4b_prefill_path_preserved(self):
        for n, k in contracts.T2_4B_NK:
            self.assertTrue(contracts.supports_layer_kernel_shape('t2g128', 1, n, k))
            self.assertFalse(contracts.supports_layer_kernel_shape('t2g128', 8, n, k))

    def test_w1a16_only_registered_head(self):
        self.assertTrue(contracts.supports_kernel_shape('w1a16', 9, 151936, 2048))
        self.assertFalse(contracts.supports_kernel_shape('w1a16', 9, 2048, 2048))

    def test_cuda_dispatch_matches_shape_contract(self):
        source = (PACKAGE / 'gpu/t2g128_kernels/t2g128_kernels.cu').read_text()
        native_m = {int(x) for x in re.findall(r'if \(M ==\s*(\d+)\)\s*\{ DISPATCH_SHAPES_4B', source)}
        self.assertEqual(native_m, contracts.T2_4B_M)
        section = source.split('#define DISPATCH_SHAPES_4B')[1].split('extern "C"')[0]
        native_nk = {(int(n), int(k)) for k, n in re.findall(r'if \(K == (\d+) && N == (\d+)\)', section)}
        self.assertEqual(native_nk, contracts.T2_4B_NK)


class DecoderContracts(unittest.TestCase):
    def test_margin_is_not_silently_relaxed(self):
        tree = ast.parse((PACKAGE / 'gpu/fast_generate.py').read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in ('fast_speculative_generate', 'fast_hybrid_generate'):
                defaults = dict(zip([x.arg for x in node.args.args][-len(node.args.defaults):], node.args.defaults))
                self.assertEqual(ast.literal_eval(defaults['rabv_margin']), 0.0)
                calls = [x for x in ast.walk(node) if isinstance(x, ast.Call)]
                self.assertTrue(any(isinstance(x.func, ast.Name) and x.func.id == 'validate_verification_options' for x in calls))

    def test_exact_options_fail_on_contradiction(self):
        contracts.validate_verification_options(0.0, True)
        for margin in (-1.0, 0.5, 1e-20, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                contracts.validate_verification_options(margin, False)
        with self.assertRaisesRegex(ValueError, 'requires rabv_margin=0'):
            contracts.validate_verification_options(0.5, True)

    def test_training_and_export_sources_included(self):
        for name in ('data.py', 'sdlm_data.py', 'train.py', 'sdlm_train.py',
                     'sdlm_modeling.py', 'sdlm_attn_mask.py', 'convert.py',
                     'binary_modules.py', 'ternary_modules.py',
                     'export_ternary.py'):
            self.assertTrue((PACKAGE / name).is_file(), name)

    def test_no_private_paths_or_checkpoint_artifacts(self):
        for path in PACKAGE.rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.cu', '.h', '.sh'):
                self.assertNotIn('/filesdir/', path.read_text(), str(path))
            # Local compile.sh builds a .so beside its source by design.
            self.assertNotIn(path.suffix, ('.bak', '.pt', '.safetensors'))


if __name__ == '__main__':
    unittest.main()
