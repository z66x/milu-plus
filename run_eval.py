"""MILU extension: log-likelihood vs direct-generation vs CoT on one stratified subsample.

Example:
  python run_eval.py --model Qwen/Qwen3-8B --modes loglik,gen,cot --n_per_cell 25
Resumable: re-run the same command after a crash / Ctrl-C and it continues.
"""
import argparse, json, os, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from common import *


def load_model(name, four_bit, attn):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(device_map={"": 0}, torch_dtype=torch.float16)
    if attn:
        kw["attn_implementation"] = attn
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
    print(f"[load] {name}  VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return tok, model


def chat_wrap(tok, text):
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    return text + "\nAnswer:"


def letter_ids(tok):
    ids = []
    for L in LETTERS:
        s = set()
        for v in (" " + L, L):
            e = tok.encode(v, add_special_tokens=False)
            if e:
                s.add(e[-1])
        ids.append(sorted(s))
    return ids


@torch.inference_mode()
def forward_last(model, enc):
    for kw in ({"logits_to_keep": 1}, {"num_logits_to_keep": 1}, {}):
        try:
            return model(**enc, **kw).logits[:, -1, :].float()
        except TypeError:
            continue


def score_loglik(tok, model, docs, k, lid):
    prompts = [loglik_prompt(d, k) for d in docs]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(model.device)
    lp = forward_last(model, enc).log_softmax(-1)
    out = []
    for i in range(len(docs)):
        sc = [torch.logsumexp(lp[i, ids], 0).item() for ids in lid]
        pred = max(range(4), key=lambda j: sc[j])
        out.append((pred, "loglik", json.dumps([round(s, 3) for s in sc]), enc["input_ids"].shape[1], pred))
    return out


@torch.inference_mode()
def score_gen(tok, model, docs, cot, max_new, lid):
    prompts = [chat_wrap(tok, gen_prompt(d, cot)) for d in docs]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
    texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    del out, enc
    parsed = [parse_answer(t, d["options"]) for d, t in zip(docs, texts)]
    fb = [p for p, _ in parsed]
    if cot:   # forced-answer fallback for truncated / unparseable generations
        bad = [i for i, p in enumerate(fb) if p is None]
        if bad:
            fp = [prompts[i] + texts[i].rstrip() + "\n\nAnswer:" for i in bad]
            e2 = tok(fp, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            lp = forward_last(model, e2).log_softmax(-1)
            for j, i in enumerate(bad):
                sc = [torch.logsumexp(lp[j, ids], 0).item() for ids in lid]
                fb[i] = max(range(4), key=lambda k: sc[k])
    return [(parsed[i][0], parsed[i][1], texts[i][:4000], len(tok.encode(texts[i])), fb[i])
            for i in range(len(docs))]


def run_batch(fn, docs):
    """Run a batch; on CUDA OOM split it in half (down to size 1)."""
    try:
        return fn(docs)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(docs) == 1:
            return [(None, "oom", "", 0, None)]
        h = len(docs) // 2
        print(f"  [oom] splitting batch {len(docs)} -> {h}+{len(docs)-h}")
        return run_batch(fn, docs[:h]) + run_batch(fn, docs[h:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--modes", default="loglik,gen,cot", help="comma list of loglik,gen,cot")
    ap.add_argument("--shots", type=int, default=0, help="few-shot k for loglik (0/1/5)")
    ap.add_argument("--n_per_cell", type=int, default=25)
    ap.add_argument("--languages", default=",".join(LANGS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", default="data/samples.json")
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only first N questions")
    ap.add_argument("--bs_loglik", type=int, default=4)
    ap.add_argument("--bs_gen", type=int, default=12)
    ap.add_argument("--bs_cot", type=int, default=12)
    ap.add_argument("--max_new_cot", type=int, default=400)
    ap.add_argument("--no4bit", action="store_true", help="fp16 (only for <=3B models)")
    ap.add_argument("--attn", default=None, help="e.g. eager for Gemma-2")
    a = ap.parse_args()

    if not os.path.exists(a.samples):
        build_samples(a.n_per_cell, a.seed, a.languages.split(","), a.samples)
    docs_all = load_samples(a.samples)
    if a.limit:                       # spread the smoke test over languages
        docs_all = docs_all[::max(1, len(docs_all) // a.limit)][:a.limit]
    print(f"[data] {len(docs_all)} questions")
    tok, model = load_model(a.model, not a.no4bit, a.attn)
    lid = letter_ids(tok)
    os.makedirs(a.out_dir, exist_ok=True)
    tag = a.model.replace("/", "_")

    for mode in a.modes.split(","):
        name = f"loglik{a.shots}" if mode == "loglik" else mode
        path = os.path.join(a.out_dir, f"{tag}__{name}{'__smoke' if a.limit else ''}.jsonl")
        done = set()
        if os.path.exists(path):
            done = {json.loads(l)["id"] for l in open(path)}
        todo = [d for d in docs_all if d["id"] not in done]
        todo.sort(key=lambda d: -len(d["question"]) - sum(map(len, d["options"])))
        bs = {"loglik": a.bs_loglik, "gen": a.bs_gen, "cot": a.bs_cot}[mode]
        if mode == "loglik":
            fn = lambda ds: score_loglik(tok, model, ds, a.shots, lid)
        else:
            fn = lambda ds: score_gen(tok, model, ds, mode == "cot", a.max_new_cot if mode == "cot" else 8, lid)
        print(f"[{name}] {len(done)} done, {len(todo)} to go, batch {bs}")
        t0 = time.time()
        with open(path, "a") as f:
            for i in range(0, len(todo), bs):
                batch = todo[i:i + bs]
                for d, (pred, how, raw, ntok, pfb) in zip(batch, run_batch(fn, batch)):
                    f.write(json.dumps({
                        "id": d["id"], "model": a.model, "mode": name, "language": d["language"],
                        "domain": d["domain"], "subject": d["subject"], "gold": d["gold"],
                        "pred": pred, "correct": int(pred == d["gold"]), "pred_fb": pfb,
                        "correct_fb": int(pfb == d["gold"]), "parse": how,
                        "raw": raw, "ntok": ntok}, ensure_ascii=False) + "\n")
                f.flush()
                el = time.time() - t0
                n = min(i + bs, len(todo))
                print(f"  {n}/{len(todo)}  {el/60:.1f} min  ETA {el/n*(len(todo)-n)/60:.0f} min", flush=True)
        torch.cuda.empty_cache()
    print("[done]")


if __name__ == "__main__":
    main()
