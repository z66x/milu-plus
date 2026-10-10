#!/usr/bin/env bash
# Overnight driver. Edit MODELS, then:  bash run_all.sh 2>&1 | tee run_all.log
# Safe to re-run: finished questions are skipped.
N=25                      # questions per language x domain (88 cells -> ~2200 questions)
MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"   # overlaps the paper -> before/after baseline
  "Qwen/Qwen3-8B"                      # newer than the paper's cutoff
)
for M in "${MODELS[@]}"; do
  # fast modes first so you have numbers early, CoT last (slowest)
  python run_eval.py --model "$M" --modes loglik,gen --n_per_cell $N
  python run_eval.py --model "$M" --modes cot        --n_per_cell $N
done
python analyze.py
