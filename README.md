# CREATEivity-benchmark

- `generation/`: generation, filtering, repair, and image-selection data/scripts.
- `experiments/`: evaluation scripts and results.
- `final_dataset/`: 155 puzzles.
- `80_human_study_puzzles/`: 80 human study puzzles.
- `human_study_responses/`: anonymized answers, ratings, and surveys.


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

## Citation

If you use the code or data from this repository, please consider citing our work:

```bibtex
@misc{yang2026create,
  title = {{CREATE}: Creativity Evaluation of {VLMs} using the Visual Remote Associates Test},
  author = {Ziqing Yang and Anubhav Jangra and Smaranda Muresan},
  year = {2026},
  eprint = {2610.18569},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2610.18569}
}
```
