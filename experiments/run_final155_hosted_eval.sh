#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
choice="${1:-}"
phase="${2:-pilot}"
case "$phase" in
  pilot) limit=1 ;;
  pilot10) limit=10 ;;
  full) limit=0 ;;
  *) printf '%s\n' 'Phase must be pilot, pilot10, or full.' >&2; exit 2 ;;
esac
settings=()
case "$choice" in
  gemini-lite) provider=gemini; model=gemini-3.5-flash-lite ;;
  gemini-flash) provider=gemini; model=gemini-3.8-flash ;;
  deepseek) provider=deepseek; model=deepseek-flash ;;
  gpt-luna) provider=openai; model=gpt-5.6-luna; settings=(--reasoning-effort none --max-output-tokens 1024) ;;
  gpt-sol) provider=openai; model=gpt-5.6-sol; settings=(--reasoning-effort none --max-output-tokens 1024) ;;
  claude-haiku) provider=anthropic; model=claude-haiku-4-5-20251001; settings=(--anthropic-thinking disabled --max-output-tokens 1024) ;;
  claude-opus) provider=anthropic; model=claude-opus-5-5; settings=(--anthropic-thinking adaptive --reasoning-effort low --max-output-tokens 2048) ;;
  *) printf '%s\n' 'Usage: bash run_final155_hosted_eval.sh MODEL {pilot|pilot10|full} [text_only|two_shot] [runner options]; see README.md.' >&2; exit 2 ;;
esac
if [[ $# -ge 2 ]]; then shift 2; elif [[ $# -eq 1 ]]; then shift; fi
condition_settings=()
output="experiments/results/new_runs/final155_${model}_v2.jsonl"
case "${1:-}" in
  text_only|two_shot)
    condition_settings=(--experiments 2 --task2-conditions "$1")
    output="experiments/results/new_runs/final155_${model}_${1}_v1.jsonl"
    shift
    ;;
  ""|--*) ;;
  *) printf '%s\n' 'Condition must be text_only or two_shot, or omitted for original tasks.' >&2; exit 2 ;;
esac
manifest=experiments/manifests/final155_manifest.jsonl
if [[ ! -f "$manifest" ]]; then
  "${PYTHON_BIN:-python3}" -B experiments/scripts/prepare_eval.py --final-dataset-jsonl final_dataset/detail/final_puzzles.jsonl --output-jsonl "$manifest"
fi
command=("${PYTHON_BIN:-python3}" -B experiments/scripts/model_evaluation.py --manifest "$manifest" --provider "$provider" --model "$model" --limit-puzzles "$limit" --timeout 600 --output-jsonl "$output" --resume)
if (( ${#settings[@]} )); then command+=("${settings[@]}"); fi
if (( ${#condition_settings[@]} )); then command+=("${condition_settings[@]}"); fi
exec "${command[@]}" "$@"
