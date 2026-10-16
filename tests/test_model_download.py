"""CPU-only download tests using local fixtures and a mocked HF client."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bonsai_sft import download_model as module


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cache = self.root / 'cache'
        self.cache.mkdir()
        self.output = self.root / 'weights'
        self.manifest = self.root / 'registry.json'
        contents = {'config.json': b'{"fixture":1}', 'tokenizer.json': b'{"fixture":2}',
                    'tokenizer_config.json': b'{"fixture":3}', 'model.safetensors': b'fixture weights',
                    'README.md': b'# Fixture model'}
        files = []
        for name, value in contents.items():
            (self.cache / name).write_bytes(value)
            files.append({'path_in_repo': name, 'size': len(value),
                          'sha256': hashlib.sha256(value).hexdigest(), 'kind': module._file_kind(name)})
        self.entry = {'repo_id': 'fixture-owner/fixture-model', 'revision': 'a' * 40, 'files': files}
        self.registry = {'schema': module.SCHEMA, 'models': {'fixture': self.entry}}
        self.save()

    def save(self):
        self.manifest.write_text(json.dumps(self.registry))

    def fake_download(self, **kwargs):
        self.assertEqual(set(kwargs), {'repo_id', 'repo_type', 'revision', 'filename'})
        self.assertEqual(kwargs['repo_id'], self.entry['repo_id'])
        self.assertEqual(kwargs['repo_type'], 'model')
        self.assertEqual(kwargs['revision'], 'a' * 40)
        return str(self.cache / kwargs['filename'])

    def run_download(self, downloader=None):
        with patch.object(module, '_hf_hub_download', side_effect=downloader or self.fake_download) as mocked:
            result = module.download_model('fixture', self.output, self.manifest)
        return result, mocked

    def assert_bad_manifest(self, pattern):
        self.save()
        with patch.object(module, '_hf_hub_download') as mocked:
            with self.assertRaisesRegex(ValueError, pattern):
                module.download_model('fixture', self.output, self.manifest)
            mocked.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_pinned_download_and_verified_receipt(self):
        result, mocked = self.run_download()
        self.assertEqual(result, self.output)
        self.assertEqual(mocked.call_count, len(self.entry['files']))
        self.assertEqual({p.name for p in self.output.iterdir()},
                         {e['path_in_repo'] for e in self.entry['files']} | {module.RECEIPT})
        receipt = json.loads((self.output / module.RECEIPT).read_text())
        self.assertEqual(receipt['revision'], 'a' * 40)
        self.assertEqual(receipt['files'], self.entry['files'])
        self.assertNotIn(str(self.root), json.dumps(receipt))

    def test_missing_registry_has_friendly_error(self):
        self.manifest.unlink()
        with self.assertRaisesRegex(ValueError, 'registry is unavailable'):
            self.run_download()
        self.assertFalse(self.output.exists())

    def test_unknown_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unknown model'):
            module.download_model('not-present', self.output, self.manifest)

    def test_existing_directory_is_untouched(self):
        self.output.mkdir()
        (self.output / 'keep').write_text('keep')
        with self.assertRaisesRegex(ValueError, 'output already exists'):
            self.run_download()
        self.assertEqual((self.output / 'keep').read_text(), 'keep')

    def test_existing_file_or_dangling_symlink_is_untouched(self):
        self.output.write_text('keep')
        with self.assertRaisesRegex(ValueError, 'output already exists'):
            self.run_download()
        self.assertEqual(self.output.read_text(), 'keep')
        self.output.unlink()
        self.output.symlink_to(self.root / 'missing')
        with self.assertRaisesRegex(ValueError, 'output already exists'):
            self.run_download()
        self.assertTrue(self.output.is_symlink())

    def test_branch_tag_and_abbreviated_revisions_are_rejected(self):
        for revision in ('main', 'v1.0', 'a' * 7, 'g' * 40, None):
            with self.subTest(revision=revision):
                self.entry['revision'] = revision
                self.assert_bad_manifest('40-character')

    def test_path_traversal_and_non_model_files_are_rejected(self):
        for name in ('../config.json', '/config.json', 'folder/config.json',
                     'optimizer.pt', 'trainer_state.json', '_bitmtp_download.json'):
            with self.subTest(name=name):
                self.entry['files'][0]['path_in_repo'] = name
                self.assert_bad_manifest('allowlist')

    def test_bad_hash_size_and_kind_are_rejected(self):
        original = copy.deepcopy(self.entry['files'][0])
        for field, value, message in (('sha256', 'unknown', 'SHA-256'), ('size', True, 'size'),
                                      ('size', 0, 'size'), ('kind', 'weights', 'kind')):
            with self.subTest(field=field, value=value):
                self.entry['files'][0] = dict(original, **{field: value})
                self.assert_bad_manifest(message)

    def test_duplicate_files_are_rejected(self):
        self.entry['files'].append(dict(self.entry['files'][0]))
        self.assert_bad_manifest('duplicate model filename')

    def test_private_or_unrecognized_file_metadata_is_rejected(self):
        self.entry['files'][0]['local_path'] = '/private/source/config.json'
        self.assert_bad_manifest('must contain only')

    def test_duplicate_json_keys_are_rejected(self):
        self.manifest.write_text('{"schema":"a", "schema":"b"}')
        with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
            self.run_download()

    def test_missing_weights_and_tokenizer_are_rejected(self):
        original = copy.deepcopy(self.entry['files'])
        for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json'):
            self.entry['files'] = [f for f in original if f['path_in_repo'] != name]
            self.assert_bad_manifest('registry must include')

    def test_sharded_weights_require_an_index(self):
        next(f for f in self.entry['files'] if f['kind'] == 'weights')['path_in_repo'] = 'model-00001-of-00001.safetensors'
        self.assert_bad_manifest('sharded models need their index')

    def sharded_registry(self, bad_index=False):
        self.entry['files'] = [f for f in self.entry['files'] if f['kind'] != 'weights']
        names = ['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']
        fixtures = {names[0]: b'first shard', names[1]: b'second shard',
                    'model.safetensors.index.json': json.dumps({'weight_map': {
                        'first': names[0], 'second': 'absent.safetensors' if bad_index else names[1]}}).encode()}
        for name, value in fixtures.items():
            (self.cache / name).write_bytes(value)
            self.entry['files'].append({'path_in_repo': name, 'size': len(value),
                                        'sha256': hashlib.sha256(value).hexdigest(),
                                        'kind': module._file_kind(name)})
        self.save()

    def test_complete_sharded_download(self):
        self.sharded_registry()
        self.run_download()
        self.assertTrue((self.output / module.RECEIPT).is_file())

    def test_missing_declared_shard_is_rejected_before_download(self):
        self.sharded_registry()
        self.entry['files'] = [f for f in self.entry['files']
                               if f['path_in_repo'] != 'model-00002-of-00002.safetensors']
        self.assert_bad_manifest('complete declared shard set')

    def test_index_with_unregistered_shard_has_no_receipt(self):
        self.sharded_registry(bad_index=True)
        with self.assertRaisesRegex(ValueError, 'index does not match'):
            self.run_download()
        self.assertFalse((self.output / module.RECEIPT).exists())

    def test_hash_mismatch_has_no_completion_receipt(self):
        (self.cache / 'model.safetensors').write_bytes(b'changed weights')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            self.run_download()
        self.assertFalse((self.output / module.RECEIPT).exists())

    def test_size_mismatch_has_no_completion_receipt(self):
        (self.cache / 'model.safetensors').write_bytes(b'truncated')
        with self.assertRaisesRegex(ValueError, 'size/type mismatch'):
            self.run_download()
        self.assertFalse((self.output / module.RECEIPT).exists())

    def test_network_failure_has_no_receipt_and_cli_does_not_print_cause(self):
        def failure(**kwargs):
            raise RuntimeError('sensitive upstream detail')
        with patch.object(module, '_hf_hub_download', side_effect=failure), patch('sys.stderr', new_callable=io.StringIO) as stderr:
            with self.assertRaises(SystemExit) as error:
                module.main(['--model', 'fixture', '--output', str(self.output), '--manifest', str(self.manifest)])
        self.assertEqual(error.exception.code, 1)
        self.assertIn('Check network and Hugging Face access', stderr.getvalue())
        self.assertNotIn('sensitive upstream detail', stderr.getvalue())
        self.assertFalse((self.output / module.RECEIPT).exists())

    def test_unexpected_output_file_prevents_completion(self):
        def downloader(**kwargs):
            (self.output / 'unexpected').write_text('unexpected')
            return self.fake_download(**kwargs)
        with self.assertRaisesRegex(ValueError, 'unexpected files'):
            self.run_download(downloader)
        self.assertFalse((self.output / module.RECEIPT).exists())

    def test_receipt_write_failure_does_not_create_completion_marker(self):
        with patch.object(module.json, 'dump', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.run_download()
        self.assertFalse((self.output / module.RECEIPT).exists())

    def test_no_token_cli_argument(self):
        with patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
            module.main(['--model', 'fixture', '--output', str(self.output), '--token', 'not-a-token'])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.output.exists())

    def test_in_memory_token_is_forwarded_but_not_persisted(self):
        ephemeral = 'synthetic-test-token'
        def downloader(**kwargs):
            self.assertEqual(kwargs.pop('token'), ephemeral)
            return self.fake_download(**kwargs)
        with patch.object(module, '_hf_hub_download', side_effect=downloader):
            module.download_model('fixture', self.output, self.manifest, token=ephemeral)
        for path in self.output.iterdir():
            self.assertNotIn(ephemeral.encode(), path.read_bytes())


if __name__ == '__main__':
    unittest.main()
