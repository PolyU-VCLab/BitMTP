"""CPU-only checks for preserving MTP metadata during HF export."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

try:
    from bonsai_sft import export_ternary
except ImportError:
    export_ternary = None


@unittest.skipIf(export_ternary is None, 'CPU torch and transformers required')
class BonsaiExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'source'
        self.source.mkdir()
        self.output = Path(self.temp.name) / 'output'
        self.saved = dict(block_size=4, causal_attn=True, text_mask_token_id=15,
                          mask_token_str='<|vision_pad|>')
        self.config = SimpleNamespace(block_size=4, causal_attn=True,
                                      text_mask_token_id=15, vocab_size=16)

    def write_config(self, value):
        (self.source / 'sdlm_config.json').write_text(json.dumps(value))

    def test_sft_without_sdlm_metadata_is_accepted(self):
        self.assertIsNone(export_ternary.validate_sdlm_config(str(self.source), self.config))

    def test_matching_d4_metadata_is_preserved_on_export(self):
        self.write_config(self.saved)
        model, tokenizer = Mock(), Mock()
        with patch.object(export_ternary.AutoConfig, 'from_pretrained', return_value=self.config), \
             patch.object(export_ternary.AutoModelForCausalLM, 'from_pretrained', return_value=model), \
             patch.object(export_ternary.AutoTokenizer, 'from_pretrained', return_value=tokenizer), \
             patch.object(export_ternary, 'convert_qwen3_to_ternary_qat'), \
             patch.object(export_ternary, 'round_latent_to_ternary'):
            export_ternary.export_round_to_safetensors(str(self.source), str(self.output))
        self.assertEqual(json.loads((self.output / 'sdlm_config.json').read_text()), self.saved)
        self.assertEqual(self.config.block_size, 4)
        model.save_pretrained.assert_called_once_with(str(self.output), safe_serialization=True)
        tokenizer.save_pretrained.assert_called_once_with(str(self.output))

    def test_conflicts_fail_before_loading_weights_or_creating_output(self):
        for key, value in (('block_size', 8), ('causal_attn', False), ('text_mask_token_id', 14)):
            with self.subTest(key=key):
                self.write_config(dict(self.saved, **{key: value}))
                with patch.object(export_ternary.AutoConfig, 'from_pretrained', return_value=self.config), \
                     patch.object(export_ternary.AutoModelForCausalLM, 'from_pretrained') as load_model:
                    with self.assertRaisesRegex(ValueError, f'conflicts.*{key}'):
                        export_ternary.export_round_to_safetensors(str(self.source), str(self.output))
                    load_model.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_invalid_or_missing_fields_are_rejected(self):
        cases = [[], dict(self.saved, block_size=True),
                 dict(self.saved, causal_attn=1), dict(self.saved, text_mask_token_id='15')]
        incomplete = dict(self.saved)
        del incomplete['block_size']
        cases.append(incomplete)
        for value in cases:
            with self.subTest(value=value):
                self.write_config(value)
                with self.assertRaises(ValueError):
                    export_ternary.validate_sdlm_config(str(self.source), self.config)

    def test_invalid_ranges_are_rejected(self):
        for field, value in (('block_size', 0), ('text_mask_token_id', 16), ('text_mask_token_id', -1)):
            with self.subTest(field=field, value=value):
                self.write_config(dict(self.saved, **{field: value}))
                config = SimpleNamespace(**vars(self.config))
                setattr(config, field, value)
                with self.assertRaises(ValueError):
                    export_ternary.validate_sdlm_config(str(self.source), config)


if __name__ == '__main__':
    unittest.main()
