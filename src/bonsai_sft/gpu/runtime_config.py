"""Explicit runtime profiles; importing this module does not import PyTorch."""
from __future__ import annotations

import sys


RUNTIME_PROFILES = ('standard', 'orin-sft', 'orin-d4', 'orin-d8')
_profile = 'standard'


def is_orin_profile(profile=None):
    return (profile if profile is not None else _profile).startswith('orin-')


def select_runtime_profile(profile):
    """Select before importing model/atlas helpers; never patch torch.compile."""
    global _profile
    if profile not in RUNTIME_PROFILES:
        raise ValueError(f'unknown runtime profile: {profile}')
    loaded = any(f'bonsai_sft.gpu.{name}' in sys.modules
                 for name in ('model', 'fast_runtime', 'fast_generate'))
    if loaded and profile != _profile:
        raise RuntimeError('select the runtime profile before importing GPU runtime modules')
    _profile = profile


def compile_helper(fn=None, **kwargs):
    """Use the same tensor expressions eagerly on Orin, compiled otherwise."""
    if is_orin_profile():
        return fn if fn is not None else lambda function: function
    import torch
    return torch.compile(fn, **kwargs) if fn is not None else torch.compile(**kwargs)


def validate_profile_options(args):
    profile = args.runtime_profile
    if profile not in RUNTIME_PROFILES:
        raise ValueError(f'unknown runtime profile: {profile}')
    if not is_orin_profile(profile):
        return
    block_size = 4 if profile == 'orin-d4' else 8
    if args.block_size != block_size:
        raise ValueError(f'{profile} requires block-size {block_size}')
    if profile == 'orin-sft' and args.mode != 'ar':
        raise ValueError('orin-sft requires AR mode')
    if profile != 'orin-sft' and args.mode == 'ar':
        raise ValueError('Orin MTP profiles require conf, spec, or hybrid mode')
    if args.draft_precision != 'native' or args.verify_precision != 'native':
        raise ValueError('Orin profiles require native T2 draft and verifier projections')
    if args.max_context != 4096:
        raise ValueError('Orin profiles require a single 4096-token segment')
    if args.threshold != 0.9:
        raise ValueError('Orin profiles require threshold 0.9')
    if args.min_chain not in (None, 3):
        raise ValueError('Orin hybrid profiles require min-chain 3')


def validate_profile_model(profile, config):
    if is_orin_profile(profile) and (config.get('dim') != 2048
                                    or config.get('kernel_kind') != 't2g128'):
        raise ValueError('Orin profiles support only Bonsai-1.7B native T2 checkpoints')


def validate_profile_device(profile, capability):
    if is_orin_profile(profile) and tuple(capability) != (8, 7):
        raise ValueError('Orin profiles require an SM87 device')
