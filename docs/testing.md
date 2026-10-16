# Tests

Run the software regression suite from the repository root in the installed development environment:

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
```

The tests cover weight packing and conversion, kernel-shape validation, verifier acceptance, causal masks, training labels, evaluation profiles, scoring, and per-task result validation. They run without a GPU and do not measure CUDA performance or generation quality.

## GPU checks

The generation CLI provides `--check-only` to validate local files and options before allocating CUDA state. With a supported model and GPU, `--exact-m1-verify` selects width-one verification for comparison with the block verifier; see [generation](usage.md).

When changing the runtime, compare token IDs, committed prefixes, and materialized causal KV on short cases covering EOS, output limits, rejection, pending-token widths, and segment transitions. Different attention tilings can produce floating-point differences, so numerical comparisons should use the same operator configuration. Verifier acceptance always requires exact token-ID agreement.

Use the [evaluation guide](evaluation.md) for task scores and throughput.
