#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Start, wait for and stop a Qwen3.5 / Qwen3.8 server (serve.sh) for the stress scripts.
#
#   srv.sh start <name> [ENV=VAL ...]   start serve.sh in the background, log to $LOG_DIR/<name>.log
#   srv.sh wait <name>                  wait until the server is up (or failed); print the result
#   srv.sh stop                         stop every running vllm serve and its engine processes
#
# Example:
#   srv.sh start 27b-v2-apc VLLM_USE_V2_MODEL_RUNNER=1 PREFIX_CACHING=1 MODEL=<ckpt> \
#       FP8_SCALES=<scales> KV_CACHE_DTYPE=fp8 BLOCK_SIZE=128 MTP_TOKENS=2
#   srv.sh wait 27b-v2-apc && python stress.py --data gsm8k_test.json gsm8k --n 200
#   srv.sh stop
#
# Environment: LOG_DIR (default ./logs), VENV (a virtualenv to activate before serve.sh;
# default: none, vllm and neuronx-cc must be on PATH).
#
# stop matches "bin/vllm serve", not "vllm serve": `pkill -f "vllm serve"` also matches
# the calling shell when its own command line contains "vllm serve" (for example
# `bash -c 'srv.sh stop; ... vllm serve ...'`) and kills it. This script's own process and
# its parent are excluded as well.
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SERVE_SH=$HERE/../serve.sh
LOG_DIR=${LOG_DIR:-$PWD/logs}

server_pids() {
  pgrep -f "bin/vllm serve" | grep -v -x -e "$$" -e "$PPID" || true
}

cmd=${1:-}
[ $# -gt 0 ] && shift
case $cmd in
stop)
  for p in $(server_pids); do kill "$p" 2>/dev/null; done
  for _ in $(seq 120); do
    pgrep -f "bin/vllm serve|VLLM::|EngineCore" | grep -v -x -e "$$" -e "$PPID" >/dev/null \
      || break
    sleep 2
  done
  for p in $(pgrep -f "VLLM::" | grep -v -x -e "$$" -e "$PPID"); do kill -9 "$p" 2>/dev/null; done
  true ;;
start)
  name=${1:?usage: srv.sh start <name> [ENV=VAL ...]}; shift
  mkdir -p "$LOG_DIR"
  log=$LOG_DIR/$name.log
  if [ -n "${VENV:-}" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
  fi
  env "$@" nohup "$SERVE_SH" > "$log" 2>&1 &
  echo "started -> $log" ;;
wait)
  name=${1:?usage: srv.sh wait <name>}
  log=$LOG_DIR/$name.log
  until grep -q -E "Application startup complete|Traceback|EngineCore failed" "$log" \
      || [ -z "$(server_pids)" ]; do
    sleep 10
  done
  grep -E -m2 "Application startup complete|Traceback|EngineCore failed|Error" "$log"
  grep -m1 "init engine" "$log" | cut -c1-200
  grep -q "Application startup complete" "$log" ;;
*)
  sed -n '4,8p' "$0"
  exit 2 ;;
esac
