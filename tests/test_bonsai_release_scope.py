"""Public CLI support policy is narrower than the internal kernel registry."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bonsai_sft.gpu.contracts import validate_public_model_scope, validate_public_device_scope
from bonsai_sft.gpu.run_generation import make_parser, validate_artifacts


HF = dict(hidden_size=2048, num_hidden_layers=28, num_attention_heads=16,
          num_key_value_heads=8, head_dim=128, intermediate_size=6144,
          vocab_size=151669, tie_word_embeddings=True)
GPU = dict(dim=2048, n_layers=28, n_heads=16, n_kv_heads=8, head_dim=128,
           ffn_dim=6144, vocab_size=151669, tie_word_embeddings=True,
           block_size=8, kernel_kind='t2g128', quantize_embedding=True)
HF4 = dict(HF, hidden_size=2560, num_hidden_layers=36, num_attention_heads=32,
           intermediate_size=9728)
GPU4 = dict(GPU, dim=2560, n_layers=36, n_heads=32, ffn_dim=9728)


class PublicReleaseScope(unittest.TestCase):
    def test_17b_d4_d8_and_sft_default_are_supported(self):
        validate_public_model_scope(HF, hf=True)  # SFT without MTP fields defaults to D8.
        for block in (4,8):
            validate_public_model_scope(dict(HF, block_size=block), hf=True)
            validate_public_model_scope(dict(GPU, block_size=block))
        legacy = dict(GPU)
        del legacy['quantize_embedding']
        validate_public_model_scope(legacy)

    def test_4b_d8_and_sft_metadata_are_supported_but_d4_is_not(self):
        validate_public_model_scope(HF4, hf=True)
        validate_public_model_scope(dict(HF4, block_size=8), hf=True)
        validate_public_model_scope(GPU4)
        for config, hf in ((dict(HF4, block_size=4), True), (dict(GPU4, block_size=4), False)):
            with self.assertRaisesRegex(ValueError, 'block_size'):
                validate_public_model_scope(config, hf=hf)

    def test_4b_gpu_scope_requires_sm90(self):
        validate_public_device_scope(GPU4, (9,0))
        validate_public_device_scope(GPU, (8,6))
        for capability in ((8,6),(8,9),(10,0)):
            with self.assertRaisesRegex(ValueError, 'requires SM90'):
                validate_public_device_scope(GPU4, capability)

    def test_4b_q1_and_unknown_dimensions_are_rejected(self):
        for update in ({'dim':2560}, {'n_layers':36}, {'n_heads':32},
                       {'n_kv_heads':4}, {'head_dim':64}, {'ffn_dim':9728},
                       {'vocab_size':151936}, {'block_size':16},
                       {'kernel_kind':'q1'}, {'quantize_embedding':False},
                       {'tie_word_embeddings':False}):
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, 'only Bonsai 1.7B T2'):
                validate_public_model_scope(dict(GPU, **update))
        with self.assertRaisesRegex(ValueError, 'only Bonsai 1.7B T2'):
            validate_public_model_scope(HF, hf=True, quant='q1')

    def test_generation_rejects_scope_before_loading_weights_or_library(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'bonsai_gpu_config.json').write_text(json.dumps(dict(GPU,dim=2560)))
            args=make_parser().parse_args(['--ckpt',str(root),'--hf-ckpt',str(root),
                                          '--block-size','8','--mode','ar','--prompt','x'])
            with self.assertRaisesRegex(ValueError, 'only Bonsai 1.7B T2'):
                validate_artifacts(args,(1,))

    def test_converter_main_keeps_internal_convert_api_separate(self):
        try:
            from bonsai_sft.gpu import convert_bonsai_mtp_to_gpu as module
        except ImportError:
            self.skipTest('CPU torch and safetensors required')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); output=root/'output'
            (root/'config.json').write_text(json.dumps(HF))
            with patch.object(module,'convert') as convert:
                module.main(['--src',str(root),'--out',str(output)])
            self.assertEqual(convert.call_args.kwargs['quant'],'ternary')
            self.assertFalse(output.exists())

    def test_converter_cli_4b_q1_and_embedding_override_write_nothing(self):
        try:
            from bonsai_sft.gpu import convert_bonsai_mtp_to_gpu as module
        except ImportError:
            self.skipTest('CPU torch and safetensors required')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); output=root/'output'
            for config,extra in ((dict(HF4,block_size=4),[]), (HF,['--quant','q1']),
                                 (HF,['--keep-embedding-fp'])):
                (root/'config.json').write_text(json.dumps(config))
                with patch.object(module,'convert') as convert, contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        module.main(['--src',str(root),'--out',str(output),*extra])
                    self.assertEqual(error.exception.code,2)
                    convert.assert_not_called()
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
