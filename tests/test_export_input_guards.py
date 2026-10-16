"""CPU-only export destination and shared MTP-metadata checks."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from bonsai_sft import export_ternary
from bonsai_sft.gpu import convert_bonsai_mtp_to_gpu as converter


class ExportInputGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'source'
        self.source.mkdir()
        self.output = Path(self.temp.name) / 'output'
        self.config = dict(
            hidden_size=128, num_hidden_layers=1, num_attention_heads=1,
            num_key_value_heads=1, head_dim=128, intermediate_size=128,
            vocab_size=16, rms_norm_eps=1e-6, rope_theta=1e6,
            rope_scaling=dict(original_max_position_embeddings=32, factor=4.0),
            tie_word_embeddings=True, block_size=4, causal_attn=True,
            text_mask_token_id=15,
        )
        self.saved = dict(block_size=4, causal_attn=True, text_mask_token_id=15,
                          mask_token_str='<|vision_pad|>')
        (self.source / 'config.json').write_text(json.dumps(self.config))

    def write_sidecar(self, value):
        (self.source / 'sdlm_config.json').write_text(json.dumps(value))

    def write_weights(self):
        weights = {'model.embed_tokens.weight': torch.ones(16, 128),
                   'model.norm.weight': torch.ones(128)}
        for name in ('input_layernorm', 'post_attention_layernorm',
                     'self_attn.q_norm', 'self_attn.k_norm'):
            weights[f'model.layers.0.{name}.weight'] = torch.ones(128)
        for name in ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                     'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'):
            weights[f'model.layers.0.{name}.weight'] = torch.ones(128, 128)
        save_file(weights, str(self.source / 'model.safetensors'))

    def assert_existing_export_rejected(self, destination):
        with patch.object(export_ternary.AutoConfig, 'from_pretrained') as config_loader, \
             patch.object(export_ternary.AutoModelForCausalLM, 'from_pretrained') as model_loader:
            with self.assertRaisesRegex(ValueError, 'export output already exists'):
                export_ternary.export_round_to_safetensors(str(self.source), str(destination))
            config_loader.assert_not_called()
            model_loader.assert_not_called()

    def test_existing_output_directory_is_preserved(self):
        self.output.mkdir()
        marker = self.output / 'keep.txt'
        marker.write_text('original')
        self.assert_existing_export_rejected(self.output)
        self.assertEqual(marker.read_text(), 'original')
        self.assertEqual(list(self.output.iterdir()), [marker])

    def test_empty_output_directory_is_rejected(self):
        self.output.mkdir()
        self.assert_existing_export_rejected(self.output)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_output_file_and_dangling_symlink_are_preserved(self):
        self.output.write_text('original')
        self.assert_existing_export_rejected(self.output)
        self.assertEqual(self.output.read_text(), 'original')
        link = Path(self.temp.name) / 'dangling'
        link.symlink_to(Path(self.temp.name) / 'absent')
        self.assert_existing_export_rejected(link)
        self.assertTrue(link.is_symlink())

    def test_export_cannot_overwrite_its_input(self):
        original = (self.source / 'config.json').read_bytes()
        self.assert_existing_export_rejected(self.source)
        self.assertEqual((self.source / 'config.json').read_bytes(), original)

    def test_shared_metadata_validator_accepts_object_and_dict(self):
        self.assertIs(converter.validate_sdlm_config, export_ternary.validate_sdlm_config)
        for block_size in (4, 8):
            saved = dict(self.saved, block_size=block_size)
            self.write_sidecar(saved)
            config = dict(self.config, block_size=block_size)
            for representation in (config, SimpleNamespace(**config)):
                self.assertEqual(export_ternary.validate_sdlm_config(str(self.source), representation), saved)

    def test_converter_rejects_conflicting_metadata_before_tensor_loading(self):
        self.write_weights()
        for field, value in (('block_size', 8), ('causal_attn', False), ('text_mask_token_id', 14)):
            with self.subTest(field=field):
                self.write_sidecar(dict(self.saved, **{field: value}))
                with patch.object(converter, 'safe_open') as tensor_reader:
                    with self.assertRaisesRegex(ValueError, f'conflicts.*{field}'):
                        converter.convert(str(self.source), str(self.output), quant='ternary')
                    tensor_reader.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_converter_rejects_invalid_sidecar_without_output(self):
        self.write_weights()
        for saved in ([], dict(self.saved, block_size=True), {'block_size': 4}):
            with self.subTest(saved=saved):
                self.write_sidecar(saved)
                with self.assertRaises(ValueError):
                    converter.convert(str(self.source), str(self.output), quant='ternary')
                self.assertFalse(self.output.exists())

    def test_matching_sidecar_preserves_converted_tensors(self):
        self.write_weights()
        without = Path(self.temp.name) / 'without-sidecar'
        converter.convert(str(self.source), str(without), quant='ternary')
        self.write_sidecar(self.saved)
        converter.convert(str(self.source), str(self.output), quant='ternary')
        for filename in ('model_state_t2g128.pt', 'model_state_fp16.pt'):
            before = torch.load(without / filename, weights_only=True)
            after = torch.load(self.output / filename, weights_only=True)
            self.assertEqual(set(before), set(after))
            for name in before:
                self.assertTrue(torch.equal(before[name], after[name]), name)
        self.assertEqual((without / 'bonsai_gpu_config.json').read_bytes(),
                         (self.output / 'bonsai_gpu_config.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
