#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
results="${1:?Usage: bash score_final155_eval.sh RESULTS_JSONL [scorer options]}"
shift
if [[ ! -f "$results" ]]; then printf 'Missing results: %s\n' "$results" >&2; exit 2; fi
stem="$(basename "$results" .jsonl)"

score_subset() {
  local subset="$1"
  shift
  local command=("${PYTHON_BIN:-python3}" -B experiments/scripts/llm_judge.py
    --manifest experiments/manifests/final155_manifest.jsonl
    --puzzles-jsonl final_dataset/detail/final_puzzles.jsonl
    --results-jsonl "$results"
    --semantic-concepts --output-dir "experiments/results/scored/${stem}_${subset}")
  if [[ "$subset" == human80 ]]; then
    command+=(--puzzle-ids-file 80_human_study_puzzles/puzzle_ids.txt)
  fi
  "${command[@]}" "$@"
}

score_subset all155 "$@"
score_subset human80 "$@"
