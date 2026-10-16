"""CPU checks for checkpoint-preserving benchmark tokenization."""
import ast
import inspect
import json
from pathlib import Path
import tempfile
import textwrap
from types import SimpleNamespace
import unittest

from tokenizers import Regex, Tokenizer, models, pre_tokenizers
from transformers import AutoTokenizer

from benchmarks.bonsai_adapter import BonsaiMTPWrap


def tokenizer_loader_call():
    tree = ast.parse(textwrap.dedent(inspect.getsource(BonsaiMTPWrap.init)))
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Name)
             and node.func.value.id == 'AutoTokenizer'
             and node.func.attr == 'from_pretrained']
    if len(calls) != 1:
        raise AssertionError('Expected one benchmark tokenizer loader')
    return calls[0]


class EvaluationTokenizerTests(unittest.TestCase):
    def test_benchmark_loader_explicitly_disables_regex_rewriting(self):
        options = {item.arg: item.value for item in tokenizer_loader_call().keywords}
        self.assertIn('fix_mistral_regex', options)
        self.assertIs(ast.literal_eval(options['fix_mistral_regex']), False)

    def test_saved_regex_is_retained_with_adjacent_model_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Exercise the large-vocabulary tokenizer-loading branch without weights.
            backend = Tokenizer(models.WordLevel(
                vocab={f't{i}': i for i in range(100001)}, unk_token='t0'))
            backend.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.Split(Regex(r'\p{L}+|\p{N}{1,3}|\s+|[^\w\s]+'),
                                     behavior='isolated')])
            saved = json.loads(backend.pre_tokenizer.__getstate__())
            backend.save(str(root / 'tokenizer.json'))
            (root / 'tokenizer_config.json').write_text(json.dumps({
                'tokenizer_class': 'PreTrainedTokenizerFast', 'unk_token': 't0',
                'model_max_length': 1024,
            }))
            (root / 'config.json').write_text(json.dumps({
                'model_type': 'qwen3', 'transformers_version': '4.57.6',
            }))
            expression = ast.Expression(tokenizer_loader_call())
            tokenizer = eval(compile(expression, '<benchmark-tokenizer-loader>', 'eval'),
                             {'AutoTokenizer': AutoTokenizer},
                             {'self': SimpleNamespace(tokenizer_path=str(root))})
            self.assertGreater(tokenizer.vocab_size, 100000)
            effective = json.loads(tokenizer.backend_tokenizer.pre_tokenizer.__getstate__())
            self.assertEqual(effective, saved)


if __name__ == '__main__':
    unittest.main()
