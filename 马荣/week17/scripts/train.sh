#!/usr/bin/env bash
set -euo pipefail
python -m grpo_math.train --config configs/grpo_qwen25_math_1_5b.yaml "$@"

