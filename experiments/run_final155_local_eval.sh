#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in
  molmo) model=allenai/Molmo2-8B; slug=molmo2-8b; eval_settings=() ;;
  qwen9b) model=Qwen/Qwen3.5-9B; slug=qwen3.5-9b_nothinking_json; eval_settings=(--local-thinking disabled --local-json-mode --max-output-tokens 512) ;;
  qwen27b) model=Qwen/Qwen3.8-27B; slug=qwen3.8-27b_nothinking_json; eval_settings=(--local-thinking disabled --local-json-mode --max-output-tokens 512) ;;
  *) source experiments/scripts/model_profiles.sh; configure_local_vlm "${1:-}" ;;
esac
phase="${2:-pilot}"
case "$phase" in
  pilot) limit=1 ;;
  pilot10) limit=10 ;;
  full) limit=0 ;;
  *) printf '%s\n' 'Phase must be pilot, pilot10, or full.' >&2; exit 2 ;;
esac
if [[ $# -ge 2 ]]; then shift 2; else shift; fi
condition_settings=(--experiments 1,2,3)
output="experiments/results/new_runs/final155_${slug}_prompt_v3.jsonl"
case "${1:-}" in
  text_only|two_shot)
    condition_settings=(--experiments 2 --task2-conditions "$1")
    output="experiments/results/new_runs/final155_${slug}_${1}_v1.jsonl"
    shift
    ;;
  ""|--*) ;;
  *) printf '%s\n' 'Condition must be text_only or two_shot, or omitted for original tasks.' >&2; exit 2 ;;
esac
# Override experiments to 3 for Task 3 only. Default covers all three tasks.
command=("${PYTHON_BIN:-python3}" -B experiments/scripts/model_evaluation.py \
  --manifest experiments/manifests/final155_manifest.jsonl --provider local --model "$model" \
  --endpoint "${EVAL_ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}" \
  --limit-puzzles "$limit" --timeout 600 \
  --output-jsonl "$output" --resume)
if (( ${#eval_settings[@]} )); then command+=("${eval_settings[@]}"); fi
command+=("${condition_settings[@]}")
exec "${command[@]}" "$@"
