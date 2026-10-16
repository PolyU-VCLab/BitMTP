"""CPU routing checks: no model/GPU initialization and no dataset download."""
import ast,contextlib,inspect,io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from benchmarks import generation
from benchmarks.profiles import PROFILES,MODEL_SPECS,task_protocol

class PlatformRoutingTests(unittest.TestCase):
    def test_check_only_never_initializes_runtime_or_cuda(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            (path/'bonsai_gpu_config.json').write_text(json.dumps(dict(MODEL_SPECS['1p7b'],kernel_kind='t2g128',block_size=4,causal_attn=True,mask_token_id=151654)))
            (path/'model_state_t2g128.pt').write_bytes(b'path fixture, not real tensors')
            (path/'tokenizer_config.json').write_text('{}')
            (path/'tokenizer.json').write_text('{}')
            argv=['generation','--profile','bonsai_1p7b_h20_d4_hybrid','--task','gsm8k',
                  '--model-id','test-model','--checkpoint',str(path),'--tokenizer',str(path),
                  '--output',str(path/'output'),'--check-only']
            output=io.StringIO()
            with patch('sys.argv',argv),patch('importlib.metadata.version',return_value='0.4.9.2'):
                with patch.object(generation,'build_adapter',side_effect=AssertionError('runtime must not build')):
                    with patch.object(torch.cuda,'device_count',side_effect=AssertionError('CUDA must not be queried')):
                        with contextlib.redirect_stdout(output):generation.main()
            result=json.loads(output.getvalue());self.assertFalse(result['gpu_executed'])
            self.assertIn('not this preflight',result['weight_tensor_validation'])
            self.assertFalse((path/'output').exists())

    def test_artifact_presence_and_content_identity(self):
        from benchmarks.contracts import artifact_files,fingerprint_artifacts,validate_artifact_identity,implementation_files_from_roots
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            for name in ('bonsai_gpu_config.json','model_state_t2g128.pt','tokenizer_config.json','tokenizer.json'):
                (path/name).write_bytes(b'fixture')
            (path/'__init__.py').write_bytes(b'')
            kernels=path/'t2g128_kernels';kernels.mkdir()
            (kernels/'libbonsai_t2.so').write_bytes(b'fake library, never loaded')
            paths=artifact_files(path,path)
            paths.update(implementation_files_from_roots(path,path))
            first=fingerprint_artifacts(paths);validate_artifact_identity(first)
            (path/'tokenizer.json').write_bytes(b'changed')
            second=fingerprint_artifacts(paths)
            self.assertNotEqual(first['tokenizer'],second['tokenizer'])
            self.assertEqual(first['model'],second['model'])
            (kernels/'libbonsai_t2.so').write_bytes(b'changed fake library')
            third=fingerprint_artifacts(paths)
            self.assertNotEqual(second['implementation'],third['implementation'])
            (kernels/'libbonsai_t2.so').unlink()
            with self.assertRaisesRegex(ValueError,'build t2g128'):implementation_files_from_roots(path,path)
            with self.assertRaisesRegex(ValueError,'local directory'):artifact_files(path,path/'absent')
            (path/'model_state_t2g128.pt').unlink()
            with self.assertRaisesRegex(ValueError,'packed model'):artifact_files(path,path)

    def test_all_profiles_route_their_own_width_and_precision(self):
        from benchmarks import bonsai_adapter
        for name,profile in PROFILES.items():
            args=SimpleNamespace(tokenizer='hf-tokenizer',checkpoint='packed-checkpoint',profile=name,task='gsm8k')
            with patch.object(bonsai_adapter.BonsaiMTPWrap,'init',return_value=None):
                wrap=generation.build_adapter(profile,args,task_protocol('gsm8k',device_profile=profile['device_profile']))
            self.assertEqual(wrap.block_size,profile['block_size'])
            self.assertEqual(wrap.min_chain,profile['min_chain'])
            self.assertEqual(wrap.replay_widths,tuple(profile['replay_widths']))
            self.assertEqual(wrap.use_bf16_draft,profile['draft_projection']=='bf16_raw')
            self.assertEqual(wrap.use_fp8_draft,profile['draft_projection']=='fp8')
            self.assertEqual(wrap.task_max_gen_tokens,{'gsm8k':1024})

    def test_actual_runtime_keywords_match_interfaces(self):
        from bonsai_sft.gpu import fast_generate
        source=Path(__file__).resolve().parents[1]/'benchmarks/bonsai_adapter.py'
        for call in ast.walk(ast.parse(source.read_text())):
            if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id in (
                'fast_ar_generate','fast_confidence_generate','fast_speculative_generate','fast_hybrid_generate'):
                self.assertFalse({k.arg for k in call.keywords}-set(inspect.signature(getattr(fast_generate,call.func.id)).parameters))

if __name__=='__main__':unittest.main()
