# Evaluation

This repository provides GSM8K and HumanEval+ evaluation for the Bonsai models. Scoring uses lm-eval and EvalPlus; BitMTP supplies the model adapter, generation settings, and runtime timers.

Evaluation uses the tokenizer as saved in the checkpoint, without regex rewriting.

## GSM8K

Run all 1,319 test examples on one H20 with a Bonsai-1.7B D8 model:

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.generation \
  --profile bonsai_1p7b_h20_d8_hybrid --task gsm8k \
  --model-id bonsai_1p7b_mtp_d8 \
  --checkpoint /path/to/packed-bonsai-d8 \
  --tokenizer /path/to/hf-bonsai-d8 \
  --output results/1p7b-d8-hybrid-gsm8k
```

The profile constructs a four-shot, multi-turn chat prompt before applying the input limit below. It uses a 1,024-token generation limit and exact match with flexible answer extraction.

## HumanEval+

HumanEval+ uses all 164 original function-completion prompts, zero-shot generation without a chat template, a 1,024-token generation limit, and Plus pass@1. Completions are cut at the configured stop strings; no additional code sanitization is applied.

Generate on the GPU host first. This command does not execute generated Python and does not require `--allow-code-execution`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.generation \
  --profile bonsai_1p7b_h20_d8_hybrid --task humaneval_plus \
  --model-id bonsai_1p7b_mtp_d8 \
  --checkpoint /path/to/packed-bonsai-d8 \
  --tokenizer /path/to/hf-bonsai-d8 \
  --output results/1p7b-d8-hybrid-humaneval-generated --generate-only
```

The new directory contains `samples.jsonl`, `request_receipts.json`, and `generation.json` with `GENERATED_UNSCORED` status. It preserves model/tokenizer/runtime identities, the actual generation hardware, raw token counts, and measured runtime statistics. It has no accuracy score or `result.json` yet. Add `--offline` when the generation host's dataset cache is already populated.

Build the CPU scorer from the repository root while network access is available. The image caches the official HumanEvalPlus-v0.1.10 dataset and installs EvalPlus 0.3.1 and CPU scoring dependencies, without Torch, GPU libraries, model weights, or tokenizers:

```bash
docker build -f docker/Dockerfile.humaneval-cpu -t bitmtp-humaneval-cpu:0.3.1 .
```

Execute generated code only in isolation. On a Linux Docker host, mount just the generated directory read-only and a separate fresh output directory:

```bash
mkdir results/1p7b-d8-hybrid-humaneval-scored
docker run --rm --runtime=runc --network=none --read-only \
  --user "$(id -u):$(id -g)" --cap-drop=ALL --security-opt=no-new-privileges \
  --cpus=8 --memory=8g --pids-limit=256 \
  --tmpfs /tmp:rw,noexec,nosuid,size=1g \
  --mount "type=bind,src=$(realpath results/1p7b-d8-hybrid-humaneval-generated),dst=/input,readonly" \
  --mount "type=bind,src=$(realpath results/1p7b-d8-hybrid-humaneval-scored),dst=/output" \
  bitmtp-humaneval-cpu:0.3.1 \
  --generation /input --output /output/run --allow-code-execution
```

Do not add GPU access, host networking, credentials, model directories, or the Docker socket to this container. `--allow-code-execution` acknowledges code execution; it does not create a sandbox. For strongly adversarial code, use a dedicated disposable VM as an additional boundary.

The scorer checks all 164 unique tasks, the fixed dataset hash, sample and request-receipt hashes, generation protocol and artifact identities, and the evaluator's exact prompt-plus-completion binding. It runs eight CPU workers with a 600-second evaluator timeout; EvalPlus's per-problem time-limit rules are unchanged. Plus correctness requires both base and Plus tests to pass.

Successful scoring writes `run/result.json` with `PASS` status, exact base/Plus correct counts, and the original GPU statistics. The CPU scoring environment is recorded separately and does not replace generation hardware. Generation and scoring wall times are separate; a two-stage run has no continuous end-to-end wall time. Failed scoring retains the generation facts and evaluator transcript without creating a passing result. Retry with a new scoring output directory; GPU regeneration is unnecessary.

The original one-process command remains available by omitting `--generate-only` and supplying `--allow-code-execution`, but the whole process must then run in an isolated environment. It also persists generation facts before scoring begins.

## Profiles and timing

`--profile` loads a built-in configuration; it does not require a JSON file. `configs/evaluation/` contains eight common profiles: H20 and RTX 3090 1.7B SFT AR and D4/D8 Hybrid, plus H20 4B SFT AR and D8 Hybrid. These files can be supplied with `--profile-file` alongside the matching `--profile`. Additional decoding modes remain available as built-in choices.

H20 profiles cover Bonsai-1.7B D4/D8 and Bonsai-4B D8, with separate `sft_ar` profiles and weights for SFT baselines. RTX 3090 profiles cover Bonsai-1.7B SFT AR, D4 Hybrid, and D8 MTP routes; MTP drafting uses BF16. See [RTX 3090](rtx3090.md).

H20 uses lm-eval 0.4.9.2; RTX 3090 uses 0.4.11. Both use EvalPlus 0.3.1. Keep model files, tokenizer, task settings, and evaluator version fixed across comparisons. See [generation](usage.md#decode) for decoding modes.

Inputs longer than 1,024 tokens retain the first token and final 1,023 tokens, which can truncate demonstration messages. Generation stops at EOS or the output limit; task stop strings are applied afterward to the text used for scoring.

Each successfully scored run writes `result.json`, request statistics, and evaluator output to a new directory. Decode throughput excludes prefill and request initialization. Runtime-request throughput includes them, but excludes tokenization, model setup, and scoring. Both measure the runtime output before task-stop postprocessing, not only the retained answer or code; they are not online-stop request throughput.

Read each task's `result.json` for its score, sample count, and decode/runtime-request throughput. Results remain separate for GSM8K and HumanEval+.
