# Third-party notices

BitMTP: When Multi-Token Prediction Meets Low-Bit Large Language Models

Copyright 2026 The BitMTP Authors.

The notices and license texts below apply to their respective components. This repository does not replace or broaden upstream licenses.

| Component or reference | Attribution and license |
| --- | --- |
| Bonsai model weights and group-128 quantization format | [Prism ML Ternary-Bonsai-1.7B](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-unpacked) and [Ternary-Bonsai-4B](https://huggingface.co/prism-ml/Ternary-Bonsai-4B-unpacked), based on Qwen3; Apache-2.0. The model downloads include the upstream `LICENSE` and `NOTICE.txt`, preserving Prism ML and Alibaba Cloud attribution. |
| Adapted low-bit GPU kernels and inference-model code in `src/bonsai_sft/gpu/` | [Microsoft BitNet / bitnet.cpp](https://github.com/microsoft/BitNet); MIT copyright notice and license in `licenses/BitNet-MIT.txt`. The xFormers-derived portions retain the separate BSD terms below. |
| xFormers-derived code in `src/bonsai_sft/gpu/model.py` | Copyright (c) Facebook, Inc. and its affiliates; BSD 3-Clause terms in `licenses/xFormers-BSD.txt` |
| SDLM/InternVL-derived training and attention-mask components in `sdlm_attn_mask.py`, `sdlm_data.py`, and `sdlm_modeling.py` | [OpenGVLab SDLM](https://github.com/OpenGVLab/SDLM); MIT copyright notice and license in `licenses/SDLM-MIT.txt` |
| Transformers/Qwen model interfaces and activation-quantization reference | [Transformers BitNet integration](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/integrations/bitnet.py); Apache-2.0. `binary_modules.py` identifies the activation-quantization reference. |
| Installed packages, including Accelerate, DeepSpeed, lm-eval and EvalPlus | Each dependency retains its own license. |

When redistributing fine-tuned or converted model weights, include the source model's license and notices and identify the fine-tuning or format conversion. Dataset terms apply separately.
