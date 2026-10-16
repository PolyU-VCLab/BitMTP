"""Single-prompt CUDA-graph generation; not a quality or speed benchmark.

Run ``python -m bonsai_sft.gpu.run_generation --help`` for the explicit
checkpoint, decoding, and role-precision options. Checkpoints and tokenizers
must already exist locally; this entry point never downloads artifacts.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .contracts import (require_kernel_shape, validate_verification_options,
                        validate_public_model_scope, validate_public_device_scope)
from .runtime_config import (RUNTIME_PROFILES, is_orin_profile, select_runtime_profile,
                             validate_profile_options, validate_profile_model,
                             validate_profile_device)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', type=Path, required=True, help='Converted GPU checkpoint directory.')
    parser.add_argument('--hf-ckpt', '--hf_ckpt', dest='hf_ckpt', type=Path, required=True,
                        help='Local Hugging Face tokenizer directory for the same checkpoint.')
    parser.add_argument('--block-size', type=int, choices=(4, 8), required=True,
                        help='Must match bonsai_gpu_config.json; not a checkpoint override.')
    parser.add_argument('--mode', choices=('ar', 'conf', 'spec', 'hybrid'), required=True)
    parser.add_argument('--runtime-profile', choices=RUNTIME_PROFILES, default='standard',
                        help='Explicit platform configuration; Orin profiles disable compilation and Flex attention.')
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument('--prompt', help='Raw prompt text, unless --chat is supplied.')
    prompt.add_argument('--prompt-file', type=Path, help='Read a UTF-8 prompt from a local file.')
    parser.add_argument('--chat', action='store_true', help='Apply the checkpoint chat template, thinking disabled.')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--max-context', type=int, default=4096,
                        help='One captured KV segment; must fit prompt + output + maximum graph width.')
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--min-chain', type=int, default=None,
                        help='Hybrid gate: default 6 for standard D8, otherwise 3. Orin fixes this at 3.')
    parser.add_argument('--margin', type=float, default=0.0,
                        help='Compatibility argument; only 0 is accepted by strict verification.')
    parser.add_argument('--exact-m1-verify', action='store_true',
                        help='Use the same width-1 graph as AR; requires margin 0.')
    parser.add_argument('--draft-precision', choices=('native', 'fp8', 'bf16'), default='native',
                        help='bf16 is the explicit draft projection for hardware without FP8, including SM86.')
    parser.add_argument('--verify-precision', choices=('native',), default='native')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-flex-attention', action='store_true')
    parser.add_argument('--check-only', action='store_true',
                        help='Validate options and local artifacts only; never initialize CUDA or generate.')
    return parser


def validate_options(args) -> tuple[tuple[int, ...], tuple[str, ...], int]:
    validate_verification_options(args.margin, args.exact_m1_verify)
    validate_profile_options(args)
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        raise ValueError('threshold must be finite and between 0 and 1')
    if args.max_new_tokens < 1 or args.max_context < 1:
        raise ValueError('max-new-tokens and max-context must be positive')
    default_chain = 3 if is_orin_profile(args.runtime_profile) or args.block_size == 4 else 6
    min_chain = args.min_chain if args.min_chain is not None else default_chain
    if not 1 <= min_chain <= args.block_size + 1:
        raise ValueError('min-chain must be in [1, block-size + 1]')
    if args.mode != 'hybrid' and args.min_chain is not None:
        raise ValueError('min-chain is only used by hybrid mode')
    if args.mode in ('ar', 'conf') and (args.exact_m1_verify or args.margin != 0 or args.verify_precision != 'native'):
        raise ValueError('verification options are only used by spec/hybrid modes')
    if args.mode == 'ar' and args.draft_precision != 'native':
        raise ValueError('AR does not use a draft projection')
    if args.verify_precision != 'native':
        raise ValueError('strict verification requires the native verifier projection')
    widths = (1,) if args.mode == 'ar' else ((1, 4, 5, 6, 7, 8, 9) if args.block_size == 4
                                           else (1, 8, 9, 10, 12, 16, 17))
    roles = ['ar']
    if args.mode != 'ar':
        roles.append({'native': 'draft', 'fp8': 'draft_fp8', 'bf16': 'draft_bf16'}[args.draft_precision])
    if args.mode in ('spec', 'hybrid'):
        roles.append('verify')
    return widths, tuple(roles), min_chain


def validate_artifacts(args, widths):
    config_path = args.ckpt / 'bonsai_gpu_config.json'
    if not config_path.is_file():
        raise ValueError(f'missing GPU checkpoint configuration: {config_path}')
    config = json.loads(config_path.read_text())
    validate_public_model_scope(config)
    validate_profile_model(args.runtime_profile, config)
    if config.get('block_size') != args.block_size:
        raise ValueError('block-size does not match the converted checkpoint configuration')
    kind = config.get('kernel_kind', 'q1')
    if kind not in ('q1', 't2g128'):
        raise ValueError(f'unsupported kernel_kind: {kind!r}')
    state = args.ckpt / ('model_state_t2g128.pt' if kind == 't2g128' else 'model_state_q1g128.pt')
    if not state.is_file():
        raise ValueError(f'missing packed checkpoint: {state}')
    if not (args.hf_ckpt / 'tokenizer_config.json').is_file():
        raise ValueError(f'missing tokenizer_config.json in {args.hf_ckpt}')
    has_tokenizer_json = (args.hf_ckpt / 'tokenizer.json').is_file()
    has_bpe_files = all((args.hf_ckpt / file).is_file() for file in ('vocab.json', 'merges.txt'))
    if not (has_tokenizer_json or has_bpe_files):
        raise ValueError(f'need tokenizer.json or both vocab.json and merges.txt in {args.hf_ckpt}')
    if args.prompt_file is not None and not args.prompt_file.is_file():
        raise ValueError(f'missing prompt file: {args.prompt_file}')
    dim = config['dim']
    q_width = config['n_heads'] * config['head_dim']
    kv_width = config['n_kv_heads'] * config['head_dim']
    body_shapes = ((q_width + 2 * kv_width, dim), (dim, q_width),
                   (2 * config['ffn_dim'], dim), (dim, config['ffn_dim']))
    for m in widths:
        for n, k in body_shapes:
            require_kernel_shape(kind, m, n, k)
        if kind == 't2g128':
            require_kernel_shape(kind, m, 151936, dim)
    if config['vocab_size'] > 151936:
        raise ValueError('checkpoint vocabulary exceeds the compiled packed-head capacity')
    kernel = Path(__file__).parent / ('t2g128_kernels/libbonsai_t2.so' if kind == 't2g128'
                                     else 'q1g128_kernels/libbonsai_q1.so')
    if not kernel.is_file():
        raise ValueError(f'missing native library: {kernel}; run its sibling compile.sh first')
    return config


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        widths, roles, min_chain = validate_options(args)
        config = validate_artifacts(args, widths)
    except (ValueError, RuntimeError, KeyError) as error:
        parser.error(str(error))
    if args.check_only:
        print(json.dumps({'status': 'local_artifacts_and_options_checked', 'gpu_executed': False,
                          'runtime_profile': args.runtime_profile,
                          'mode': args.mode, 'kernel_kind': config.get('kernel_kind', 'q1'),
                          'widths': widths, 'roles': roles, 'margin': args.margin,
                          'min_chain': min_chain if args.mode == 'hybrid' else None}, indent=2))
        return

    select_runtime_profile(args.runtime_profile)
    import torch
    from transformers import AutoTokenizer
    from .generate import build_model
    from .fast_runtime import GraphAtlas
    from .fast_generate import (fast_ar_generate, fast_confidence_generate,
                                fast_speculative_generate, fast_hybrid_generate)

    if not torch.cuda.is_available():
        parser.error('CUDA is required for generation; --check-only performs artifact checks without CUDA')
    device = torch.device(args.device)
    if device.type != 'cuda':
        parser.error('device must be a CUDA device')
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    try:
        validate_public_device_scope(config, capability)
        validate_profile_device(args.runtime_profile, capability)
    except ValueError as error:
        parser.error(str(error))
    if args.draft_precision == 'fp8' and capability < (8, 9):
        parser.error('FP8 draft projection requires compatible hardware; choose native or BF16 draft')
    tokenizer = AutoTokenizer.from_pretrained(str(args.hf_ckpt), local_files_only=True,
                                              trust_remote_code=False)
    prompt = args.prompt_file.read_text() if args.prompt_file else args.prompt
    if args.chat:
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                               tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    if input_ids.shape[1] == 0:
        parser.error('prompt must tokenize to at least one token')
    if input_ids.shape[1] + args.max_new_tokens + max(widths) > args.max_context:
        parser.error('max-context is too small for prompt + output + maximum graph width')
    input_ids = input_ids.to(device)
    model = build_model(str(args.ckpt), use_kernel=True, device=device)
    model.prepare_merged_weights(max_positions=args.max_context + max(widths))
    atlas = GraphAtlas(model, model.args, device, segment_ladder=(args.max_context,),
                       m_set=widths, roles=roles,
                       use_flex_attention=not args.no_flex_attention and not is_orin_profile())
    common = dict(max_gen_len=args.max_new_tokens)
    verify = dict(use_fp8_draft=args.draft_precision == 'fp8',
                  use_bf16_draft=args.draft_precision == 'bf16',
                  exact_m1_verify=args.exact_m1_verify, rabv_margin=args.margin)
    if args.mode == 'ar':
        output, stats = fast_ar_generate(model, atlas, input_ids, **common)
    elif args.mode == 'conf':
        output, stats = fast_confidence_generate(model, atlas, input_ids, **common,
                                                 threshold=args.threshold,
                                                 use_fp8_draft=args.draft_precision == 'fp8',
                                                 use_bf16_draft=args.draft_precision == 'bf16')
    elif args.mode == 'spec':
        output, stats = fast_speculative_generate(model, atlas, input_ids, **common, **verify)
    else:
        output, stats = fast_hybrid_generate(model, atlas, input_ids, **common, **verify,
                                             threshold=args.threshold, hybrid_m=min_chain)
    print(json.dumps({'mode': args.mode, 'block_size': args.block_size, 'margin': args.margin,
                      'runtime_profile': args.runtime_profile,
                      'threshold': args.threshold, 'min_chain': min_chain if args.mode == 'hybrid' else None,
                      'roles': roles, 'exact_m1_verify': args.exact_m1_verify,
                      'text': tokenizer.decode(output.tolist(), skip_special_tokens=True),
                      'token_ids': output.tolist(), 'generation_stats': stats,
                      'note': 'Single-prompt generation only; not a quality or speed benchmark.'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
