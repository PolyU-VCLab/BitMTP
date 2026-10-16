"""Check the documented synthetic JSONL record through both real datasets."""
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import torch
except ImportError:
    torch = None


class ChatTokenizerFixture:
    """Minimal Qwen-style markers; no model or tokenizer download."""
    eos_token_id = 151645
    pad_token_id = 151643
    role_ids = {'user': 872, 'assistant': 77091, 'system': 8948}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and add_generation_prompt is False and enable_thinking is False
        return json.dumps(messages)

    @classmethod
    def tokens_and_labels(cls, messages):
        tokens, labels, assistant_content_positions = [], [], []
        for message in messages:
            role = message['role']
            header = [151644, cls.role_ids[role], 198]
            tokens.extend(header)
            labels.extend([-100] * len(header))
            offset = 2000 if role == 'assistant' else 1000
            content = [offset + ord(char) for char in message['content']]
            if role == 'assistant':
                assistant_content_positions.extend(range(len(tokens), len(tokens) + len(content)))
            tokens.extend(content + [cls.eos_token_id, 198])
            labels.extend(content + [cls.eos_token_id, -100] if role == 'assistant'
                          else [-100] * (len(content) + 2))
        return tokens, labels, assistant_content_positions

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return self.tokens_and_labels(json.loads(text))[0]

    def __call__(self, text, *, return_tensors, padding, truncation, max_length, add_special_tokens):
        assert return_tensors == 'np' and padding is False and truncation is True
        tokens = self.encode(text, add_special_tokens=add_special_tokens)
        return {'input_ids': np.asarray([tokens[:max_length]], dtype=np.int64)}


@unittest.skipIf(torch is None, 'CPU torch and numpy are required')
class TrainingJsonlExampleTests(unittest.TestCase):
    def setUp(self):
        doc = Path(__file__).resolve().parents[1] / 'docs/training.md'
        examples = re.findall(r'```jsonl\n(.*?)\n```', doc.read_text(), re.DOTALL)
        self.assertEqual(len(examples), 1)
        lines = examples[0].splitlines()
        self.assertEqual(len(lines), 1)
        self.record = json.loads(lines[0])
        self.messages = self.record['messages']
        self.assertEqual([m['role'] for m in self.messages], ['user', 'assistant'])
        self.assertTrue(all(isinstance(m['content'], str) and m['content'] for m in self.messages))
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.data = Path(temp.name) / 'example.jsonl'
        self.data.write_text(lines[0] + '\n', encoding='utf-8')
        self.tokenizer = ChatTokenizerFixture()
        self.ids, self.labels, self.assistant_positions = self.tokenizer.tokens_and_labels(self.messages)

    def test_sft_accepts_documented_record_and_masks_prompt(self):
        from bonsai_sft.data import BonsaiSFTDataset
        dataset = BonsaiSFTDataset(str(self.data), self.tokenizer, max_length=512)
        self.assertEqual(len(dataset), 1)
        sample = dataset[0]
        self.assertEqual(sample['input_ids'].tolist(), self.ids)
        self.assertEqual(sample['labels'].tolist(), self.labels)
        supervised = sample['labels'][sample['labels'] != -100].tolist()
        self.assertEqual(supervised, [self.ids[p] for p in self.assistant_positions] + [151645])

    def test_mtp_accepts_record_and_keeps_both_losses_in_assistant_spans(self):
        from bonsai_sft.sdlm_data import BonsaiSDLMDataset
        for block_size in (4, 8):
            with self.subTest(block_size=block_size):
                dataset = BonsaiSDLMDataset(str(self.data), self.tokenizer, block_size=block_size,
                                            max_length=512, expected_mask_repeat_times=2)
                self.assertEqual(len(dataset), 1)
                try:
                    with patch('bonsai_sft.sdlm_data.random.randint', return_value=0):
                        sample = dataset[0]
                finally:
                    if dataset._fh is not None:
                        dataset._fh.close()
                n = sample['x0_len'].item()
                self.assertEqual(n, len(self.ids))
                self.assertEqual(sample['input_ids'][:n].tolist(), self.ids)
                self.assertEqual(sample['labels'][:n].tolist(), self.labels)
                self.assertEqual(sample['labels'][n].item(), -100)
                block_inputs = sample['input_ids'][n:-1].tolist()
                anchors = sample['position_ids'][n:-1:block_size].tolist()
                self.assertTrue(anchors)
                self.assertTrue(set(anchors) <= set(self.assistant_positions))
                self.assertEqual(block_inputs[::block_size], [self.ids[p] for p in anchors])
                self.assertTrue(all(token == 151654 for i, token in enumerate(block_inputs)
                                    if i % block_size))
                mtp_labels = sample['labels'][n + 1:].tolist()
                valid = [token for token in mtp_labels if token != -100]
                self.assertTrue(valid)
                self.assertTrue(set(valid) <= {self.ids[p] for p in self.assistant_positions} | {151645})


if __name__ == '__main__':
    unittest.main()
