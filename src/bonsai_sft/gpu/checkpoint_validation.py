"""Read-only HF checkpoint metadata checks before Bonsai conversion writes."""
from __future__ import annotations

import json
import math
from pathlib import Path

from safetensors import safe_open


def _positive_number(value, name, *, integer=False):
    valid_type = type(value) is int if integer else type(value) in (int, float)
    if not valid_type or not math.isfinite(value) or value <= 0:
        raise ValueError(f'config.json: {name} must be a positive {"integer" if integer else "number"}')


def expected_tensor_shapes(config):
    """Core Qwen3/Bonsai tensors consumed by the GPU model, including all layers."""
    dim = config['hidden_size']
    vocab = config['vocab_size']
    head = config['head_dim']
    query = config['num_attention_heads'] * head
    kv = config['num_key_value_heads'] * head
    ffn = config['intermediate_size']
    expected = {'model.embed_tokens.weight': (vocab, dim), 'model.norm.weight': (dim,)}
    if not config['tie_word_embeddings']:
        expected['lm_head.weight'] = (vocab, dim)
    layer_shapes = {
        'input_layernorm.weight': (dim,),
        'post_attention_layernorm.weight': (dim,),
        'self_attn.q_proj.weight': (query, dim),
        'self_attn.k_proj.weight': (kv, dim),
        'self_attn.v_proj.weight': (kv, dim),
        'self_attn.o_proj.weight': (dim, query),
        'self_attn.q_norm.weight': (head,),
        'self_attn.k_norm.weight': (head,),
        'mlp.gate_proj.weight': (ffn, dim),
        'mlp.up_proj.weight': (ffn, dim),
        'mlp.down_proj.weight': (dim, ffn),
    }
    for layer in range(config['num_hidden_layers']):
        expected.update({f'model.layers.{layer}.{key}': shape for key, shape in layer_shapes.items()})
    return expected


def validate_hf_checkpoint(src_ckpt_dir):
    """Return config and shard names after checking container/key/shape metadata.

    This does not inspect tensor values or establish model quality/parity. It
    never creates an output directory or downloads/converts source artifacts.
    """
    source = Path(src_ckpt_dir)
    config_path = source / 'config.json'
    if not config_path.is_file():
        raise ValueError(f'missing HF config.json: {config_path}')
    try:
        config = json.loads(config_path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f'cannot read valid HF config.json: {config_path}') from error
    if not isinstance(config, dict):
        raise ValueError('config.json must contain a JSON object')
    for key in ('hidden_size', 'num_hidden_layers', 'num_attention_heads',
                'num_key_value_heads', 'head_dim', 'intermediate_size', 'vocab_size'):
        _positive_number(config.get(key), key, integer=True)
    for key in ('rms_norm_eps', 'rope_theta'):
        _positive_number(config.get(key), key)
    if type(config.get('tie_word_embeddings')) is not bool:
        raise ValueError('config.json: tie_word_embeddings must be an explicit boolean')
    rope = config.get('rope_scaling')
    if not isinstance(rope, dict):
        raise ValueError('config.json: Bonsai rope_scaling metadata is required')
    _positive_number(rope.get('original_max_position_embeddings'),
                     'rope_scaling.original_max_position_embeddings', integer=True)
    _positive_number(rope.get('factor'), 'rope_scaling.factor')
    if config['num_attention_heads'] % config['num_key_value_heads']:
        raise ValueError('config.json: attention heads must be divisible by key/value heads')
    if config['head_dim'] % 2:
        raise ValueError('config.json: head_dim must be even for rotary embedding')
    for name, width in (('hidden_size', config['hidden_size']),
                        ('intermediate_size', config['intermediate_size']),
                        ('query projection width', config['num_attention_heads'] * config['head_dim'])):
        if width % 128:
            raise ValueError(f'config.json: {name} must be divisible by group size 128')
    _positive_number(config.get('block_size', 8), 'block_size', integer=True)
    if type(config.get('causal_attn', True)) is not bool:
        raise ValueError('config.json: causal_attn must be a boolean')
    mask = config.get('text_mask_token_id', 151654)
    if type(mask) is not int or not 0 <= mask < config['vocab_size']:
        raise ValueError('config.json: text_mask_token_id must index the checkpoint vocabulary')

    shards = sorted(path for path in source.glob('*.safetensors') if path.is_file())
    if not shards:
        raise ValueError(
            'No HF safetensors weight shards found. A config/card or .bin-only '
            'checkpoint is not sufficient; export the complete HF model with '
            'save_pretrained(..., safe_serialization=True) before conversion.'
        )
    expected = expected_tensor_shapes(config)
    allowed = dict(expected)
    if config['tie_word_embeddings']:
        allowed['lm_head.weight'] = (config['vocab_size'], config['hidden_size'])
    seen = set()
    for shard in shards:
        try:
            with safe_open(str(shard), framework='pt', device='cpu') as reader:
                keys = list(reader.keys())
                if not keys:
                    raise ValueError(f'empty safetensors shard: {shard.name}')
                for key in keys:
                    if key in seen:
                        raise ValueError(f'duplicate tensor key across shards: {key}')
                    if key not in allowed:
                        raise ValueError(f'unsupported tensor key for Bonsai GPU conversion: {key}')
                    tensor = reader.get_slice(key)
                    shape = tuple(tensor.get_shape())
                    if shape != allowed[key]:
                        raise ValueError(f'tensor shape mismatch for {key}: {shape} != {allowed[key]}')
                    if tensor.get_dtype() not in ('F16', 'BF16', 'F32', 'F64'):
                        raise ValueError(f'{key} must contain floating-point HF weights, not packed/integer weights')
                    seen.add(key)
        except Exception as error:
            raise ValueError(f'invalid HF safetensors shard {shard.name}: {error}') from error
    missing = sorted(set(expected) - seen)
    if missing:
        raise ValueError(f'incomplete HF checkpoint: missing {len(missing)} core tensors: {missing[:5]}')
    return config, [path.name for path in shards]
