# Training

The examples train Bonsai-1.7B with SFT followed by joint next-token and block-prediction training. The saved SFT checkpoint initializes MTP. Training uses your own processed JSONL data and local model weights.

## Data format

Use one JSON object per line with a `messages` list. Each message contains string-valued `role` and `content` fields. For example, this fictional exchange is a complete JSONL record:

```jsonl
{"messages":[{"role":"user","content":"Suggest a name for a fictional lantern shop."},{"role":"assistant","content":"Moonberry Lanterns would be a cheerful name."}]}
```

Both loaders apply the model's chat template with thinking disabled. Only assistant content and its closing end-of-turn token are supervised; user messages and role headers are masked. Include an assistant response and choose a sequence length that retains it. MTP additionally forms block-prediction targets within assistant responses. Use data you are authorized to train on and retain its applicable attribution and terms.

## Train

Install the training dependencies with `python -m pip install -e '.[training]'`. For a 1.7B SFT run:

```bash
python scripts/train_bonsai.py \
  --config configs/training/sft_example.json \
  --model /path/to/base-hf-model --data /path/to/train.jsonl \
  --output /path/to/new-sft-output --nproc-per-node 8 --dry-run
```

For 1.7B D4 MTP, initialize from the saved HF SFT checkpoint, not a packed runtime directory:

```bash
python scripts/train_bonsai.py \
  --config configs/training/mtp_example.json \
  --model /path/to/sft-hf-checkpoint --data /path/to/train.jsonl \
  --output /path/to/new-mtp-output --nproc-per-node 8 --dry-run
```

`--dry-run` prints the launch command without training; inspect it before removing the flag. The launcher derives gradient accumulation from the configured global batch and worker count. Use a new output directory: it starts a new optimizer and scheduler rather than resuming an interrupted run.

The MTP example uses `block_size: 4`. For D8, copy the example beside it and set `block_size` to `8`; the remaining example settings are unchanged. Keep `deepspeed_zero2.json` beside the configuration.

For additional options:

```bash
python -m bonsai_sft.train --help
python -m bonsai_sft.sdlm_train --help
```

## Export

Use `bonsai_sft.export_ternary` for a rounded Hugging Face export and `bonsai_sft.gpu.convert_bonsai_mtp_to_gpu` for the packed runtime files described in [usage](usage.md).

Before sharing a trained or exported checkpoint, include the input model's license and notices and describe your fine-tuning or conversion in its model card; the training and export commands do not copy these notices automatically.
