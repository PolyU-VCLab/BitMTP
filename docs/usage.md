# Conversion and generation

Use trusted local weights and the tokenizer belonging to the same model. The release configurations cover T2-g128 Bonsai-1.7B at D4/D8 and Bonsai-4B at D8. The 4B runtime configuration targets H20.

## Download weights

For private Hugging Face models, run `hf auth login` with an account that has access.

```bash
python -m bonsai_sft.download_model \
  --model bonsai_1p7_d8 --output weights/BitMTP-Bonsai-1.7B-D8
```

Available model keys are `bonsai_1p7_sft`, `bonsai_1p7_d4`, `bonsai_1p7_d8`, `bonsai_4b_sft`, and `bonsai_4b_d8`. Use a new output directory. The source weights are latent QAT parameters, not prepacked ternary inference weights; use the conversion step below rather than an unmodified Transformers loader.

## Convert weights

The source directory must contain `config.json`, complete safetensors weights, and matching tokenizer files. The output path must not exist; conversion will not overwrite an earlier model.

```bash
python -m bonsai_sft.gpu.convert_bonsai_mtp_to_gpu \
  --src /path/to/hf-bonsai --out /path/to/packed-bonsai --quant ternary
```

The converted directory records the architecture and block configuration in `bonsai_gpu_config.json`. For MTP models, the block size must match training. An SFT export uses a default block-size field; this does not mean it was MTP-trained. Retain both the packed and floating-point files produced by conversion. Do not combine files from different model versions.

Before sharing converted weights, copy the source model's `LICENSE` and `NOTICE.txt` into the output directory and state that the weights were converted with BitMTP; conversion does not copy these files automatically.

## Decode

Jetson Orin uses separate native-only runtime profiles and a minimum hybrid chain of 3; see [Jetson Orin](orin.md).

For D8 hybrid decoding on H20:

```bash
python -m bonsai_sft.gpu.run_generation \
  --ckpt /path/to/packed-bonsai --hf-ckpt /path/to/hf-bonsai \
  --block-size 8 --mode hybrid --threshold 0.9 --min-chain 6 \
  --draft-precision fp8 --verify-precision native \
  --chat --prompt 'Write a short greeting.' --max-new-tokens 128
```

For 1.7B D4, use D4 weights, `--block-size 4`, and `--min-chain 3`. The block size must match the converted model configuration. The CLI supports `ar`, `conf`, `spec`, and `hybrid`:

| Mode | Draft projection on H20 | Additional options |
| --- | --- | --- |
| `ar` | None; use `--draft-precision native` | Use the corresponding SFT model for the paper's AR baseline |
| `conf` | `fp8` | `--threshold 0.9`; omit `--min-chain` |
| `spec` | `native` | Omit `--min-chain` |
| `hybrid` | `fp8` | `--threshold 0.9`; minimum chain 3 for D4 or 6 for D8 |

The SFT artifact uses `block_size=8` in its packed metadata. With the direct CLI, use `--block-size 8 --mode ar --draft-precision native`; AR still processes one token at a time. The evaluation suite provides a separate `sft_ar` profile for this baseline.

`spec` verifies every draft block with native T2 and exact token-ID matching. `conf` commits a confidence-selected prefix. `hybrid` commits that prefix when it reaches `--min-chain`; otherwise, it verifies the block.

`--check-only` validates model files and options without allocating CUDA state.

See [evaluation](evaluation.md) for benchmark prompts, stopping rules, and timing.
