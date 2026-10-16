#!/usr/bin/env bash
# Source this file from the launch and evaluation helpers.
configure_local_vlm() {
  server_settings=()
  eval_settings=(--local-json-mode --max-output-tokens 1024)
  case "$1" in
    gemma12b-it)
      model=google/gemma-4-12B-it
      slug=gemma-4-12b-it
      default_gpus=0
      eval_settings+=(--local-thinking disabled)
      server_settings+=(--reasoning-parser gemma4)
      ;;
    internvl8b)
      model=OpenGVLab/InternVL3_5-8B-HF
      slug=internvl3_5-8b-hf
      default_gpus=0
      ;;
    scout-int4)
      model=RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16
      slug=llama4-scout-instruct-w4a16
      default_gpus=0,2
      ;;
    scout-instruct)
      model=meta-llama/Llama-4-Scout-17B-16E-Instruct
      slug=llama4-scout-instruct-bf16
      default_gpus=
      ;;
    *)
      printf '%s\n' 'Choose gemma12b-it, internvl8b, scout-int4, or scout-instruct.' >&2
      return 2
      ;;
  esac
}
