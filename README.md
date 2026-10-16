# CREATEivity-benchmark

- `generation/`: generation, filtering, repair, and image-selection data/scripts.
- `experiments/`: evaluation scripts and saved results.
- `final_dataset/final_dataset_155/`: 155 puzzles and 465 images.
- `80_human_study_puzzles/`: 80 study puzzles and 240 images.
- `human_study_responses/`: anonymized answers, ratings, and structured surveys.

Use Python 3.10+ with `requirements.txt`; local inference requires GPUs and vLLM.
Git LFS is required to download the large frequency-data file.
Run generation scripts from `generation/`. Experiment commands run from the package root.
Replace `MODEL` with a launcher alias (e.g., `qwen9b`), `MODEL_ID` with its checkpoint ID,
and `OUTPUT` with your output folder. Adjust GPU settings for your model.

```bash
# Server terminal
CUDA_VISIBLE_DEVICES=0 vllm serve MODEL_ID --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 32768 --max-num-batched-tokens 8192 --max-num-seqs 1 --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image":9,"video":0}'
```

```bash
# Intended word recognition
bash experiments/run_final155_local_eval.sh MODEL full --experiments 1 --output-jsonl OUTPUT/words.jsonl

# Shared concept identification
for setting in zeroshot detailed stepwise two_shot text_only; do
  bash experiments/run_final155_local_eval.sh MODEL full --experiments 2 --task2-conditions "$setting" --max-input-images 9 --output-jsonl "OUTPUT/$setting.jsonl" || break
done

# Word recognition with cues
bash experiments/run_final155_local_eval.sh MODEL full --experiments 3 --output-jsonl OUTPUT/cues.jsonl

# Score both puzzle sets
for name in words zeroshot detailed stepwise two_shot text_only cues; do
  bash experiments/score_final155_eval.sh "OUTPUT/$name.jsonl"
done
```
