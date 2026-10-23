#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# OpenAI-compatible server for Qwen3.8-Flash-Next (qwen4_exp) on trn2.48xlarge: one TP=8 replica on
# 8 logical NeuronCores, FP8 routed experts. Up to 8 replicas fit the instance; run one per core
# range and port, and put dp_proxy.py in front (see run_replicas.sh).
#
# Usage:
#   MODEL=<path>/Qwen3.8-Flash-Next CORES=0-7 PORT=8001 examples/vllm_neuron/models/qwen4_exp/serve.sh
#
# Knobs:
#   MAX_MODEL_LEN (default 65536), MAX_NUM_SEQS (8), PREFILL_BUCKETS ("1024, 4096": short prompts
#   pad to the smallest, long ones are chunked by the largest), SEQS_BUCKETS ("1, 8"),
#   DECODE_CTX_BUCKETS ("4096, 16384, 32768": decode reads only the bucket's KV; MAX_MODEL_LEN is
#   the implicit last one), GPU_MEM_UTIL (0.82: 16.5 GiB weights + ~3.2 GiB KV per 24 GiB core; the QSA indexer views share the KV pool and the graph I/O check NCC_EVRF009 counts it twice), KV_CAP_FRACTION (0.5), SERVED_MODEL_NAME
#   (qwen3.8-flash-next).
#
# The n-gram embedding rows are built on the host from the sampled tokens every step
# (HasHostInputs); async scheduling joins on the sampled tokens before each step (host-token join).
set -euo pipefail

MODEL=${MODEL:?set MODEL to the checkpoint directory}
CORES=${CORES:-0-7}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
PREFILL_BUCKETS=${PREFILL_BUCKETS:-1024, 4096}
LARGEST_PREFILL=${PREFILL_BUCKETS##*,}; LARGEST_PREFILL=${LARGEST_PREFILL// /}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}

first=${CORES%-*}; last=${CORES#*-}
VISIBLE=$(seq -s, "${first}" "${last}")
export NEURON_VISIBLE_DEVICES=${VISIBLE}             # logical core ids
export NEURON_SKIP_EFA_AFFINITY=1
export VLLM_NEURON_BACKEND=${BACKEND:-${VLLM_NEURON_BACKEND:-vllm_neuron}}
export NEURON_LIBTORCH_COMPILATION_TIMEOUT=${NEURON_LIBTORCH_COMPILATION_TIMEOUT:-3600}
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=${KV_CAP_FRACTION:-0.5}

NEURON_CONFIG="\"quantization\": \"fp8\""
NEURON_CONFIG="${NEURON_CONFIG}, \"num_batched_tokens_buckets\": [${PREFILL_BUCKETS}]"
NEURON_CONFIG="${NEURON_CONFIG}, \"kv_segment_size_buckets\": [${PREFILL_BUCKETS}]"
NEURON_CONFIG="${NEURON_CONFIG}, \"num_seqs_buckets\": [${SEQS_BUCKETS:-1, ${MAX_NUM_SEQS}}]"
if [[ -n "${DECODE_CTX_BUCKETS:-}" ]]; then
  NEURON_CONFIG="${NEURON_CONFIG}, \"decode_context_length_buckets\": [${DECODE_CTX_BUCKETS}]"
fi
if [[ -n "${DECODE_INTERLEAVE:-}" ]]; then  # decode steps between the chunks of a long prefill
  NEURON_CONFIG="${NEURON_CONFIG}, \"decode_steps_between_prefill_chunks\": ${DECODE_INTERLEAVE}"
fi

exec vllm serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME:-qwen3.8-flash-next}" \
  --port "${PORT}" \
  --tensor-parallel-size 8 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${LARGEST_PREFILL}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.82}" \
  --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --additional-config "{\"neuron_config\": {${NEURON_CONFIG}}}"
