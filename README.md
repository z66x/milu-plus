# MILU-Plus: Log-likelihood vs. Generation vs. Chain-of-Thought on MILU

Course project, Machine Learning, IIT Bhilai (Phase 2 status).

**Team:** Tanmay Modi (B24CS047) · Kanishk Nandeshwar (B24DS010) · S Aparajith (B24DS507)

## What this project is

MILU (Multi-task Indic Language Understanding, Verma et al., 2025) is a ~80K-question multiple-choice benchmark
over 11 Indic languages, 8 domains and 41 subjects. The original paper scored open models by **log-likelihood only**,
scored API models by **generation**, and never used **chain-of-thought (CoT)**. No model was evaluated under both
protocols, so part of the reported language and domain gaps could be an artifact of the scoring method.

We extend the paper along the two axes it flags as limited:

1. **Evaluation protocol:** score the *same* models on the *same* questions with log-likelihood, direct generation
   and CoT, and measure paired differences.
2. **Model coverage:** add newer models that fit on a single free T4 GPU (≤ 9B parameters, 4-bit).

## Results (preliminary)

Sample: 1,320 questions, stratified at 15 per language × domain (11 languages × 8 domains), identical for every row.
Accuracy in %.

| Model | Protocol | Avg | High-res | Low-res | STEM | Culture |
|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | Log-likelihood (0-shot) | 42.7 | 56.7 | 39.5 | 44.1 | 39.0 |
| | Direct generation | 43.1 | 56.2 | 40.2 | 45.0 | 41.4 |
| | Chain-of-thought | 46.4 | 57.5 | 43.9 | 48.0 | 41.6 |
| Qwen3-8B | Log-likelihood (0-shot) | 55.5 | 61.7 | 54.2 | 61.5 | 48.5 |
| | Direct generation | 54.2 | 62.9 | 52.2 | 58.9 | 48.1 |
| | Chain-of-thought | 62.3 | 69.2 | 60.7 | 65.9 | 56.8 |

Paired differences on the same questions (percentage points, 95% bootstrap CI):

| Model | Generation − log-lik | CoT − log-lik |
|---|---|---|
| Qwen2.5-7B-Instruct | +0.5 [−2.1, +3.0] | +3.7 [+0.9, +6.7] |
| Qwen3-8B | −1.4 [−3.6, +0.9] | +6.7 [+4.0, +9.4] |

**Findings so far**

- Switching from log-likelihood to direct generation does not change accuracy detectably.
- Letting the model reason (CoT) improves accuracy significantly for both models (+3.7 and +6.7 pp).
- Qwen3-8B outperforms Qwen2.5-7B by about 13–16 pp, and the high- vs. low-resource gap shrinks from about 14–17 pp to about 8–11 pp.
- Low-resource languages and culture-heavy domains still lag in every configuration.
- CoT outputs were unparseable in < 1% of cases; the strict and fallback-assisted accuracies differ by ≤ 0.4 pp.

**Definitions and caveats**

- *High-res* = English + Hindi (our grouping, 240 questions); *Low-res* = the other nine languages.
- *STEM* = Science, Engineering & Technology, Health & Medicine, Environmental Sciences.
  *Culture* = Arts & Humanities, Law & Governance, Social Sciences. Business Studies is in neither group.
- With ~120 questions per language, per-language numbers are noisy; read trends, not rankings.
- CoT reasoning is requested in English even for non-English questions (Indic scripts use many more tokens, which
  would truncate reasoning and penalize low-resource languages). Qwen3 is run with thinking mode disabled so that
  CoT is our prompt, not the model's native mode.
- Neither Qwen model appears in the original paper's results, so these numbers are not compared with published
  ones. A paper-covered baseline is on the roadmap.

## Method

| Mode | What happens |
|---|---|
| `loglik0` | Single forward pass; compare the model's log-probability of the letters A/B/C/D. 0-shot (`--shots 5` for 5-shot). Equivalent in spirit to the paper's protocol, not byte-identical to `lm_eval`'s MILU task. |
| `gen` | The model is asked to answer with only a letter. |
| `cot` | Brief step-by-step reasoning (≤ 6 sentences, in English), ending with `Answer: X`. |

Answers from `gen` / `cot` are parsed with a regex parser (`common.parse_answer`). `correct` is strict (unparseable
counts as wrong); `correct_fb` additionally gives truncated generations a forced-answer fallback (letter
log-probability after the partial reasoning). Both are reported by `analyze.py`.

Models are loaded in 4-bit (bitsandbytes NF4) with fp16 compute. Batches that run out of GPU memory are split
automatically, and runs are resumable (finished question IDs are skipped).

## Repository layout

```
common.py        schema detection, stratified sampler, prompts, answer parser
run_eval.py      loads a model and runs loglik / gen / cot on the shared sample
analyze.py       aggregation, paired deltas with bootstrap CIs, plots
check_setup.py   GPU / dataset sanity check
test_common.py   parser tests
results/         summary tables and plots
runs/            per-question predictions (raw model text removed)
run_all.sh       template for batch runs (edit the model list before use)
```

## Reproducing

MILU is a gated dataset: request access at https://huggingface.co/datasets/ai4bharat/MILU first. The sample is not
redistributed here; it is regenerated deterministically (seed 0, 15 per language × domain).

```bash
pip install -r requirements.txt
huggingface-cli login
python check_setup.py            # verifies GPU and MILU column names
python run_eval.py --model Qwen/Qwen3-8B --modes loglik,gen,cot --n_per_cell 15 \
    --samples data/samples_n15.json --bs_loglik 8 --bs_gen 16 --bs_cot 16
python analyze.py                # writes results/
```

Run every model with the same `--samples` file so comparisons stay paired. Our runs used free Colab T4 GPUs
(≈ 1–1.5 h per model for all three modes).

## Status and roadmap

- [x] Stratified sampler over all 11 languages × 8 domains
- [x] Log-likelihood, direct-generation and CoT evaluation with answer parser
- [x] Paired analysis with confidence intervals
- [x] Two models evaluated (Qwen2.5-7B-Instruct, Qwen3-8B)
- [ ] Paper-covered baseline (Llama-3.1-8B-Instruct or Gemma-2-9B-it) with 5-shot log-likelihood
- [ ] Larger sample (30–50 questions per cell) and 0/1/5-shot log-likelihood for all models
- [ ] Additional post-cutoff model
- [ ] Port the generation/CoT task into `lm-evaluation-harness` (`generate_until` + answer filter)
- [ ] Heatmaps of the CoT effect by language × domain, error analysis, significance tests on gaps
- [ ] Final report

## AI assistance disclosure

The code scaffolding, prompt design and analysis scripts were developed with assistance from Claude (Anthropic).
The team ran all experiments on Google Colab T4 GPUs and spot-checked model outputs and parser behavior.
All prompts used for the generation and CoT protocols are in `common.py` (`gen_prompt`).

## References

- S. Verma, M. S. U. R. Khan, V. Kumar, R. Murthy, J. Sen. *MILU: A Multi-task Indic Language Understanding Benchmark.* NAACL 2025.
- L. Gao et al. *A framework for few-shot language model evaluation* (lm-evaluation-harness), 2021.
- J. Wei et al. *Chain-of-thought prompting elicits reasoning in large language models.* 2022.
