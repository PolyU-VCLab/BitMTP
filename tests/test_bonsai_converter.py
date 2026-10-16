"""Synthetic CPU fixtures only: conversion must reject incomplete HF inputs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
    from safetensors.torch import save_file
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'CPU torch and safetensors are required')
class BonsaiConverterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / 'hf'
        self.output = Path(self.tmp.name) / 'converted'
        self.source.mkdir()
        self.config = {
            'hidden_size': 128, 'num_hidden_layers': 1, 'num_attention_heads': 1,
            'num_key_value_heads': 1, 'head_dim': 128, 'intermediate_size': 128,
            'vocab_size': 16, 'rms_norm_eps': 1e-6, 'rope_theta': 1e6,
            'rope_scaling': {'original_max_position_embeddings': 32, 'factor': 4.0},
            'tie_word_embeddings': True, 'block_size': 4, 'causal_attn': True,
            'text_mask_token_id': 15,
        }
        self.write_config()

    def write_config(self):
        (self.source / 'config.json').write_text(json.dumps(self.config))

    def weights(self):
        # Deliberately tiny stand-in shapes, never real model weights.
        result = {'model.embed_tokens.weight': torch.ones(16, 128),
                  'model.norm.weight': torch.ones(128)}
        prefix = 'model.layers.0.'
        for name in ('input_layernorm', 'post_attention_layernorm',
                     'self_attn.q_norm', 'self_attn.k_norm'):
            result[prefix + name + '.weight'] = torch.ones(128)
        for name in ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                     'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'):
            result[prefix + name + '.weight'] = torch.ones(128, 128)
        return result

    def convert(self, **options):
        from bonsai_sft.gpu.convert_bonsai_mtp_to_gpu import convert
        return convert(str(self.source), str(self.output), **options)

    def assert_rejected_without_output(self, pattern, **options):
        with self.assertRaisesRegex(ValueError, pattern):
            self.convert(**options)
        self.assertFalse(self.output.exists(), 'invalid source created conversion artifacts')

    def test_missing_config_writes_nothing(self):
        (self.source / 'config.json').unlink()
        self.assert_rejected_without_output('missing HF config.json')

    def test_bin_only_requires_explicit_safetensors_export(self):
        (self.source / 'pytorch_model.bin').write_bytes(b'not model weights')
        self.assert_rejected_without_output('safe_serialization=True')

    def test_invalid_container_writes_nothing(self):
        (self.source / 'model.safetensors').write_bytes(b'not safetensors')
        self.assert_rejected_without_output('invalid HF safetensors shard')

    def test_empty_shard_writes_nothing(self):
        save_file({}, str(self.source / 'model.safetensors'))
        self.assert_rejected_without_output('empty safetensors shard')

    def test_missing_core_tensor_writes_nothing(self):
        weights = self.weights()
        del weights['model.layers.0.self_attn.k_proj.weight']
        save_file(weights, str(self.source / 'model.safetensors'))
        self.assert_rejected_without_output('incomplete HF checkpoint.*missing 1 core')

    def test_missing_layer_is_rejected(self):
        self.config['num_hidden_layers'] = 2
        self.write_config()
        save_file(self.weights(), str(self.source / 'model.safetensors'))
        self.assert_rejected_without_output('incomplete HF checkpoint.*missing 11 core')

    def test_wrong_tensor_shape_writes_nothing(self):
        weights = self.weights()
        weights['model.layers.0.self_attn.q_proj.weight'] = torch.ones(127, 128)
        save_file(weights, str(self.source / 'model.safetensors'))
        self.assert_rejected_without_output('tensor shape mismatch')

    def test_packed_or_integer_hf_input_rejected(self):
        weights = self.weights()
        weights['model.embed_tokens.weight'] = torch.ones(16, 128, dtype=torch.int8)
        save_file(weights, str(self.source / 'model.safetensors'))
        self.assert_rejected_without_output('floating-point HF weights')

    def test_duplicate_tensor_keys_rejected(self):
        weights = self.weights()
        save_file(weights, str(self.source / 'model-1.safetensors'))
        save_file({'model.norm.weight': torch.ones(128)}, str(self.source / 'model-2.safetensors'))
        self.assert_rejected_without_output('duplicate tensor key across shards')

    def test_unexpected_tensor_keys_rejected(self):
        weights = self.weights()
        weights['unrecognized.weight'] = torch.ones(1)
        save_file(weights, str(self.source / 'model.safetensors'))
        self.assert_rejected_without_output('unsupported tensor key')

    def test_untied_model_requires_lm_head(self):
        self.config['tie_word_embeddings'] = False
        self.write_config()
        save_file(self.weights(), str(self.source / 'model.safetensors'))
        self.assert_rejected_without_output('incomplete HF checkpoint.*lm_head')

    def test_bad_config_rejected_before_output(self):
        save_file(self.weights(), str(self.source / 'model.safetensors'))
        original = copy.deepcopy(self.config)
        for key, bad in (('rope_scaling', None), ('hidden_size', 127),
                         ('tie_word_embeddings', 'true'), ('text_mask_token_id', 100)):
            with self.subTest(key=key):
                self.config = copy.deepcopy(original)
                self.config[key] = bad
                self.write_config()
                self.assert_rejected_without_output('config.json')

    def test_invalid_source_does_not_change_existing_output(self):
        self.output.mkdir()
        (self.output / 'sentinel.txt').write_text('keep')
        with self.assertRaises(ValueError):
            self.convert()
        self.assertEqual([p.name for p in self.output.iterdir()], ['sentinel.txt'])
        self.assertEqual((self.output / 'sentinel.txt').read_text(), 'keep')

    def test_valid_source_cannot_overwrite_existing_output(self):
        save_file(self.weights(), str(self.source / 'model.safetensors'))
        self.output.mkdir()
        sentinel = self.output / 'model_state_t2g128.pt'
        sentinel.write_bytes(b'previous checkpoint')
        with self.assertRaisesRegex(ValueError, 'output already exists'):
            self.convert(quant='ternary')
        self.assertEqual(sentinel.read_bytes(), b'previous checkpoint')
        self.assertEqual([p.name for p in self.output.iterdir()], [sentinel.name])

    def test_existing_empty_output_directory_is_rejected(self):
        save_file(self.weights(), str(self.source / 'model.safetensors'))
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, 'output already exists'):
            self.convert(quant='ternary')
        self.assertEqual(list(self.output.iterdir()), [])

    def test_existing_file_or_dangling_symlink_is_rejected(self):
        self.output.write_bytes(b'keep')
        with self.assertRaisesRegex(ValueError, 'output already exists'):
            self.convert(quant='ternary')
        self.assertEqual(self.output.read_bytes(), b'keep')
        self.output.unlink()
        self.output.symlink_to(self.output.parent / 'missing')
        with self.assertRaisesRegex(ValueError, 'output already exists'):
            self.convert(quant='ternary')
        self.assertTrue(self.output.is_symlink())

    def test_invalid_output_or_grid_options_write_nothing(self):
        self.assert_rejected_without_output('at least one output', pack_kernel=False, pack_fp16=False)
        self.assert_rejected_without_output('Q1-only', quant='ternary', quantize_embedding=False)
        self.assert_rejected_without_output('quant must be', quant='not-a-grid')

    def test_valid_q1_synthetic_conversion_preserves_dtypes(self):
        save_file(self.weights(), str(self.source / 'model.safetensors'))
        self.convert(quant='q1', quantize_embedding=False)
        packed = torch.load(self.output / 'model_state_q1g128.pt', weights_only=True)
        frozen = torch.load(self.output / 'model_state_fp16.pt', weights_only=True)
        key = 'layers.0.attention.q_proj'
        self.assertEqual(packed[key + '.weight'].dtype, torch.int8)
        self.assertEqual(tuple(packed[key + '.weight'].shape), (128, 16))
        self.assertEqual(packed[key + '.weight_scale'].dtype, torch.bfloat16)
        self.assertEqual(packed['embed_tokens.weight'].dtype, torch.bfloat16)
        self.assertEqual(frozen[key + '.weight'].dtype, torch.bfloat16)
        cfg = json.loads((self.output / 'bonsai_gpu_config.json').read_text())
        self.assertEqual(cfg['kernel_kind'], 'q1')
        self.assertFalse(cfg['quantize_embedding'])

    def test_valid_ternary_synthetic_conversion_preserves_dtypes(self):
        save_file(self.weights(), str(self.source / 'model.safetensors'))
        self.convert(quant='ternary')
        packed = torch.load(self.output / 'model_state_t2g128.pt', weights_only=True)
        key = 'layers.0.attention.q_proj'
        self.assertEqual(packed[key + '.weight'].dtype, torch.int8)
        self.assertEqual(tuple(packed[key + '.weight'].shape), (128, 32))
        self.assertEqual(packed[key + '.weight_scale'].dtype, torch.bfloat16)
        self.assertEqual(packed['norm.weight'].dtype, torch.bfloat16)
        self.assertEqual(packed['embed_tokens.weight'].dtype, torch.bfloat16)
        cfg = json.loads((self.output / 'bonsai_gpu_config.json').read_text())
        self.assertEqual(cfg['kernel_kind'], 't2g128')
        self.assertEqual(cfg['block_size'], 4)

    def test_4b_metadata_uses_query_width_not_hidden_width(self):
        from bonsai_sft.gpu.checkpoint_validation import expected_tensor_shapes
        config = dict(self.config, hidden_size=2560, num_attention_heads=32,
                      num_key_value_heads=8, intermediate_size=9728)
        shapes = expected_tensor_shapes(config)
        self.assertEqual(shapes['model.layers.0.self_attn.q_proj.weight'], (4096, 2560))
        self.assertEqual(shapes['model.layers.0.self_attn.k_proj.weight'], (1024, 2560))
        self.assertEqual(shapes['model.layers.0.self_attn.o_proj.weight'], (2560, 4096))
        self.assertEqual(shapes['model.layers.0.mlp.down_proj.weight'], (2560, 9728))

if __name__ == '__main__':
    unittest.main()
