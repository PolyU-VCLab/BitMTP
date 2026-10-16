"""Download a pinned model snapshot and verify its registered files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

SCHEMA = 'bitmtp_model_registry_v1'
RECEIPT = '_bitmtp_download.json'
DEFAULT_MANIFEST = Path(__file__).with_name('model_registry.json')
CONFIG_FILES = {'config.json', 'generation_config.json', 'sdlm_config.json',
                'model.safetensors.index.json'}
TOKENIZER_FILES = {'added_tokens.json', 'chat_template.jinja', 'merges.txt',
                   'special_tokens_map.json', 'tokenizer.json',
                   'tokenizer_config.json', 'vocab.json'}
METADATA_FILES = {'README.md', 'LICENSE', 'LICENSE.md', 'LICENSE.txt',
                  'NOTICE', 'NOTICE.md', 'NOTICE.txt', '.gitattributes'}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _file_kind(name):
    if name == 'model.safetensors' or re.fullmatch(r'model-\d{5}-of-\d{5}\.safetensors', name):
        return 'weights'
    if name in CONFIG_FILES:
        return 'config'
    if name in TOKENIZER_FILES:
        return 'tokenizer'
    if name in METADATA_FILES:
        return 'metadata'
    raise ValueError(f'file is outside the model download allowlist: {name!r}')


def load_model(manifest, model):
    """Validate one model's immutable revision and exact download allowlist."""
    path = Path(manifest)
    if not path.is_file():
        raise ValueError('Model registry is unavailable. Install a release containing '
                         'model_registry.json or provide --manifest with a complete registry.')
    data = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique_object)
    if not isinstance(data, dict) or data.get('schema') != SCHEMA:
        raise ValueError(f'model registry must use schema {SCHEMA!r}')
    models = data.get('models')
    if not isinstance(models, dict) or model not in models:
        raise ValueError(f'unknown model {model!r}; inspect the registry for supported model names')
    entry = models[model]
    if not isinstance(entry, dict):
        raise ValueError('model entry must be an object')
    repo_id = entry.get('repo_id')
    if not isinstance(repo_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*', repo_id):
        raise ValueError('repo_id must be a Hugging Face owner/model name')
    revision = entry.get('revision')
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-fA-F]{40}', revision):
        raise ValueError('revision must be an exact 40-character hexadecimal commit ID')
    files = entry.get('files')
    if not isinstance(files, list) or not files:
        raise ValueError('model files must be a nonempty list')
    names = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError('each model file must be an object')
        if set(item) != {'path_in_repo', 'size', 'sha256', 'kind'}:
            raise ValueError('model files must contain only path_in_repo, size, sha256, and kind')
        name = item.get('path_in_repo')
        if not isinstance(name, str):
            raise ValueError('each model file needs path_in_repo')
        if name in names:
            raise ValueError(f'duplicate model filename: {name}')
        names.add(name)
        if item.get('kind') != _file_kind(name):
            raise ValueError(f'incorrect file kind for {name}')
        if type(item.get('size')) is not int or item['size'] <= 0:
            raise ValueError(f'file size must be a positive integer: {name}')
        if not isinstance(item.get('sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', item['sha256']):
            raise ValueError(f'file SHA-256 must be 64 lowercase hexadecimal characters: {name}')
    required = {'config.json', 'tokenizer.json', 'tokenizer_config.json'}
    if not required <= names or not any(f['kind'] == 'weights' for f in files):
        raise ValueError('registry must include weights, config.json, tokenizer.json, and tokenizer_config.json')
    shards = [f for f in files if f['kind'] == 'weights']
    if any(f['path_in_repo'] != 'model.safetensors' for f in shards):
        if 'model.safetensors' in names or 'model.safetensors.index.json' not in names:
            raise ValueError('sharded models need their index and cannot include an unsharded model file')
        numbers = [tuple(map(int, re.fullmatch(r'model-(\d{5})-of-(\d{5})\.safetensors',
                                              f['path_in_repo']).groups())) for f in shards]
        if {total for _, total in numbers} != {len(shards)} or {number for number, _ in numbers} != set(range(1, len(shards) + 1)):
            raise ValueError('model shard names must enumerate the complete declared shard set')
    return entry


def _hf_hub_download(**kwargs):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError('Install huggingface_hub before downloading models.') from error
    return hf_hub_download(**kwargs)


def _verify_file(path, expected):
    if not path.is_file() or path.is_symlink() or path.stat().st_size != expected['size']:
        raise ValueError(f'downloaded file size/type mismatch: {path.name}')
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(chunk)
    if checksum.hexdigest() != expected['sha256']:
        raise ValueError(f'downloaded file SHA-256 mismatch: {path.name}')


def download_model(model, output, manifest=DEFAULT_MANIFEST, *, token=None):
    """Use standard HF authentication; a receipt exists only after all checks pass.

    Failed downloads leave an incomplete directory without a receipt. Inspect it
    before removing it or choose a fresh output directory for another attempt.
    """
    target = Path(output).expanduser()
    if token is not None and (not isinstance(token, str) or not token):
        raise ValueError('an explicit authentication token must be a nonempty string')
    if os.path.lexists(target):
        raise ValueError(f'output already exists; choose a fresh directory: {target}')
    entry = load_model(manifest, model)
    target.mkdir(parents=True, exist_ok=False)
    for item in entry['files']:
        name = item['path_in_repo']
        try:
            request = dict(repo_id=entry['repo_id'], repo_type='model',
                           revision=entry['revision'], filename=name)
            if token is not None:
                request['token'] = token
            cached = _hf_hub_download(**request)
        except Exception as error:
            raise RuntimeError(f'Could not download {name} from {entry["repo_id"]}. '
                               f'Check network and Hugging Face access. Incomplete output: {target}') from error
        with Path(cached).open('rb') as source, (target / name).open('xb') as dest:
            shutil.copyfileobj(source, dest, length=8 * 1024 * 1024)
        _verify_file(target / name, item)
    expected = {item['path_in_repo'] for item in entry['files']}
    if {p.name for p in target.iterdir()} != expected:
        raise ValueError('download directory contains unexpected files; no completion receipt was written')
    if 'model.safetensors.index.json' in expected:
        index = json.loads((target / 'model.safetensors.index.json').read_text(encoding='utf-8'),
                           object_pairs_hook=_unique_object)
        mapping = index.get('weight_map') if isinstance(index, dict) else None
        shard_names = {f['path_in_repo'] for f in entry['files'] if f['kind'] == 'weights'}
        if (not isinstance(mapping, dict) or not mapping or
                not all(isinstance(value, str) for value in mapping.values()) or
                set(mapping.values()) != shard_names):
            raise ValueError('downloaded weight index does not match the complete registered shard set')
    receipt = {'schema': 'bitmtp_verified_download_v1', 'model': model,
               'repo_id': entry['repo_id'], 'revision': entry['revision'],
               'files': entry['files']}
    receipt_temp = target / (RECEIPT + '.tmp')
    with receipt_temp.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    receipt_temp.replace(target / RECEIPT)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='model name from the release registry')
    parser.add_argument('--output', required=True, help='new local directory for this model')
    parser.add_argument('--manifest', default=str(DEFAULT_MANIFEST), help='model registry JSON path')
    args = parser.parse_args(argv)
    try:
        path = download_model(args.model, args.output, args.manifest)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'error: {error}\n')
    print(f'Verified model files saved to {path}')


if __name__ == '__main__':
    main()
