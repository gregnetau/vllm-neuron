#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# OpenAI-compatible server for Qwen3.5 / Qwen3.8 dense (27B), TP=4 on trn2.3xlarge.
#
# Usage:
#   MODEL=<path-to-checkpoint>/Qwen3.8-27B examples/vllm_neuron/models/qwen3_5/serve.sh
#   # Static FP8 MLPs:
#   MODEL=... FP8_SCALES=<path>/fp8_act_amax.json examples/vllm_neuron/models/qwen3_5/serve.sh
#   # neuronx-cc optimization level (default 1; 3 speeds up prefill, longer compile):
#   MODEL=... OPTLEVEL=3 examples/vllm_neuron/models/qwen3_5/serve.sh
#   # FP8 KV cache (unit KV scales):
#   MODEL=... KV_CACHE_DTYPE=fp8 examples/vllm_neuron/models/qwen3_5/serve.sh
#   # Prefix caching (state checkpoints per 1024-token block; PREFILL_BUCKET >= 1024):
#   MODEL=... PREFIX_CACHING=1 examples/vllm_neuron/models/qwen3_5/serve.sh
#   # MTP speculative decoding with N (1 or 2) draft tokens:
#   MODEL=... MTP_TOKENS=1 examples/vllm_neuron/models/qwen3_5/serve.sh
#   # 32k context (KV/state pool of about 360k tokens with FP8 weights):
#   MODEL=... FP8_SCALES=... MTP_TOKENS=2 MAX_MODEL_LEN=32768 GPU_MEM_UTIL=0.9 KV_CAP_FRACTION=0.35 \
#       examples/vllm_neuron/models/qwen3_5/serve.sh
#
# Other knobs: PREFILL_BUCKET (default 1024), MAX_NUM_SEQS (default 1; larger batches produce
# wrong output, see the recipe), DECODE_CTX_BUCKETS, ASYNC_SCHEDULING (default 1),
# SERVED_MODEL_NAME (default: the checkpoint directory name).
set -euo pipefail

MODEL=${MODEL:?set MODEL to the checkpoint directory}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
PREFILL_BUCKET=${PREFILL_BUCKET:-1024}

export NEURON_SKIP_EFA_AFFINITY=1                   # trn2.3xlarge has no EFA
# KV/state pool = min(free HBM, KV_CAP_FRACTION * GPU_MEM_UTIL * 24 GB) per logical core.
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=${KV_CAP_FRACTION:-0.15}

# Decode batch buckets (each is a compiled graph); e.g. SEQS_BUCKETS="1, 4, 8" with MAX_NUM_SEQS=8.
NEURON_CONFIG="\"num_batched_tokens_buckets\": [${PREFILL_BUCKET}], \"num_seqs_buckets\": [${SEQS_BUCKETS:-${MAX_NUM_SEQS:-1}}]"
if [[ -n "${DECODE_CTX_BUCKETS:-}" ]]; then  # e.g. "4096, 16384, 32768": decode reads only the needed KV
  NEURON_CONFIG="${NEURON_CONFIG}, \"decode_context_length_buckets\": [${DECODE_CTX_BUCKETS}]"
fi
if [[ -n "${HLO2TENSORIZER_OPTIONS+set}" ]]; then  # "" = whole-graph compilation (no modular flow)
  NEURON_CONFIG="${NEURON_CONFIG}, \"hlo2tensorizer_options\": \"${HLO2TENSORIZER_OPTIONS}\""
fi
if [[ -n "${FP8_SCALES:-}" ]]; then
  NEURON_CONFIG="${NEURON_CONFIG}, \"quantization\": \"fp8\", \"fp8_activation_scales_path\": \"${FP8_SCALES}\""
fi

SPEC_ARGS=()
if [[ -n "${MTP_TOKENS:-}" ]]; then
  SPEC_ARGS=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${MTP_TOKENS}}")
fi
[[ "${ASYNC_SCHEDULING:-1}" == 1 ]] || SPEC_ARGS+=(--no-async-scheduling)

exec vllm serve "${MODEL}" "${SPEC_ARGS[@]}" \
  --served-model-name "${SERVED_MODEL_NAME:-$(basename "${MODEL}")}" \
  --tensor-parallel-size 4 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS:-1}" \
  --max-num-batched-tokens "${PREFILL_BUCKET}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.65}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-auto}" \
  --optimization-level "${OPTLEVEL:-1}" \
  "$( [[ "${PREFIX_CACHING:-0}" == 1 ]] && echo --enable-prefix-caching || echo --no-enable-prefix-caching )" \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --additional-config "{\"neuron_config\": {${NEURON_CONFIG}}}"
