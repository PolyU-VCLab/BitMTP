"""Assistant-only training must not silently become whole-sequence training."""
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import torch
except ImportError:
    torch = None


class _Tokenizer:
    eos_token_id = 151645
    pad_token_id = 151643

    def __init__(self, ids):
        self.ids = list(ids)

    def apply_chat_template(self, messages, **kwargs):
        return 'fixture'

    def encode(self, text, **kwargs):
        return list(self.ids)

    def __call__(self, text, *, max_length, **kwargs):
        return {'input_ids': np.array([self.ids[:max_length]], dtype=np.int64)}


@unittest.skipIf(torch is None, 'CPU torch and numpy are required')
class AssistantSupervisionTests(unittest.TestCase):
    ids = [151644, 872, 198, 21, 151645, 198,
           151644, 77091, 198, 31, 32, 151645]
    messages = [{'role': 'user', 'content': 'question'},
                {'role': 'assistant', 'content': 'answer'}]

    def sft_sample(self, ids, max_length=64):
        from bonsai_sft.data import BonsaiSFTDataset
        dataset = BonsaiSFTDataset.__new__(BonsaiSFTDataset)
        dataset.records = [{'messages': self.messages}]
        dataset.tokenizer = _Tokenizer(ids)
        dataset.max_length = max_length
        dataset.enable_thinking = False
        return dataset[0]

    def sdlm_sample(self, ids, max_length=64):
        from bonsai_sft.sdlm_data import preprocess_bonsai_sdlm
        # Fix only this fixture's anchor jitter; production sampling is unchanged.
        with patch('bonsai_sft.sdlm_data.random.randint', return_value=0):
            return preprocess_bonsai_sdlm(self.messages, _Tokenizer(ids), block_size=4,
                                          max_length=max_length)

    def test_sdlm_missing_markers_rejects_full_sequence_fallback(self):
        for ids in ([11, 12, 13, 14], self.ids[:6], [], [151644, 77091, 999, 31]):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, 'refusing whole-sequence'):
                self.sdlm_sample(ids)

    def test_truncated_away_assistant_rejected(self):
        with self.assertRaisesRegex(ValueError, 'refusing whole-sequence'):
            self.sdlm_sample(self.ids, max_length=8)
        with self.assertRaisesRegex(ValueError, 'no supervised assistant tokens'):
            self.sft_sample(self.ids, max_length=8)

    def test_sft_all_ignored_sample_rejected(self):
        for ids in ([11, 12, 13], self.ids[:6], []):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, 'no supervised assistant tokens'):
                self.sft_sample(ids)

    def test_recognized_assistant_sft_labels_unchanged(self):
        sample = self.sft_sample(self.ids)
        self.assertEqual(sample['labels'].tolist(), [-100] * 9 + [31, 32, 151645])

    def test_recognized_assistant_sdlm_labels_and_anchors_unchanged(self):
        sample = self.sdlm_sample(self.ids)
        self.assertEqual(sample['x0_len'].item(), 12)
        self.assertEqual(sample['labels'][:12].tolist(), [-100] * 9 + [31, 32, 151645])
        self.assertEqual(sample['labels'][12].item(), -100)  # bridge
        self.assertEqual(sample['position_ids'][12].item(), 9)  # assistant anchor
        self.assertEqual(sample['labels'][13:].tolist(), [32, 151645, -100, -100])

    def test_existing_partial_assistant_span_remains_supported(self):
        sample = self.sft_sample(self.ids, max_length=11)
        self.assertEqual(sample['labels'].tolist(), [-100] * 9 + [31, 32])
        sample = self.sdlm_sample(self.ids, max_length=11)
        self.assertEqual(sample['labels'][:11].tolist(), [-100] * 9 + [31, 32])

    def test_collators_keep_user_and_pad_labels_ignored(self):
        from bonsai_sft.data import BonsaiSFTCollator
        from bonsai_sft.sdlm_data import BonsaiSDLMCollator
        batch = BonsaiSFTCollator()([self.sft_sample(self.ids), self.sft_sample(self.ids[:11])])
        self.assertTrue((batch['labels'][:, :9] == -100).all())
        self.assertEqual(batch['labels'][1, -1].item(), -100)
        batch = BonsaiSDLMCollator()([self.sdlm_sample(self.ids), self.sdlm_sample(self.ids[:11])])
        self.assertTrue((batch['labels'][:, :9] == -100).all())
        self.assertEqual(batch['labels'][1, -1].item(), -100)


if __name__ == '__main__':
    unittest.main()
