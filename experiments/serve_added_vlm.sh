#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source experiments/scripts/model_profiles.sh
configure_local_vlm "${1:-}"
shift
if [[ "$model" == meta-llama/* && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  printf '%s\n' 'BF16 Scout needs about 218 GB for weights alone. Set CUDA_VISIBLE_DEVICES for a sufficiently large, allocated GPU group.' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$default_gpus}"
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
tp="${TENSOR_PARALLEL_SIZE:-${#gpu_ids[@]}}"
if [[ "$model" == RedHatAI/* && "$tp" -lt 2 ]]; then
  printf '%s\n' 'For Scout W4A16 on A100 40GB, allocate at least two GPUs; memory fit still requires a pilot.' >&2
  exit 2
fi
command=("${VLLM_BIN:-vllm}" serve "$model" --host 127.0.0.1 --port "${PORT:-8000}"
  --dtype bfloat16 --tensor-parallel-size "$tp" --max-model-len "${MAX_MODEL_LEN:-32768}"
  --max-num-batched-tokens 8192 --max-num-seqs 1 --gpu-memory-utilization 0.90
  --limit-mm-per-prompt "{\"image\":${MAX_INPUT_IMAGES:-3},\"video\":0}")
if (( ${#server_settings[@]} )); then command+=("${server_settings[@]}"); fi
command+=("$@")
printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
printf '%q ' "${command[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" == 1 ]]; then exit 0; fi
if ! command -v "${VLLM_BIN:-vllm}" >/dev/null; then
  printf '%s\n' 'vllm not found. Activate your puzzle-vllm environment first.' >&2
  exit 2
fi
exec "${command[@]}"
