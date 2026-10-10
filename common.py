"""Shared helpers: schema normalisation, stratified sampling, prompts, answer parser."""
import json, os, random, re

LANGS = ["Bengali", "English", "Gujarati", "Hindi", "Kannada", "Malayalam",
         "Marathi", "Odia", "Punjabi", "Tamil", "Telugu"]
LETTERS = "ABCD"
DATASET = "ai4bharat/MILU"


def _first(doc, names):
    for n in names:
        if n in doc and doc[n] is not None:
            return doc[n]
    return None


def _gold_index(raw, options):
    """Turn whatever the dataset stores as the answer into an index 0..3."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if 0 <= raw <= 3 else (raw - 1 if raw == 4 else None)
    s = str(raw).strip()
    m = re.fullmatch(r"(?i)option[\s_]*([1-4])", s)
    if m:
        return int(m.group(1)) - 1
    if re.fullmatch(r"[A-Da-d]", s):
        return LETTERS.index(s.upper())
    if re.fullmatch(r"[0-3]", s):
        return int(s)
    for i, o in enumerate(options):          # answer stored as option text
        if str(o).strip() == s:
            return i
    return None


def normalize(doc, language, idx):
    q = _first(doc, ["question", "Question", "prompt", "input"])
    opts = None
    for key in ("options", "choices"):
        if isinstance(doc.get(key), (list, tuple)) and len(doc[key]) >= 4:
            opts = list(doc[key])[:4]
    if opts is None:
        for pat in ("option{}", "option_{}", "Option{}", "choice{}"):
            cand = [doc.get(pat.format(i)) for i in range(1, 5)]
            if all(c is not None for c in cand):
                opts = cand
                break
    if opts is None:
        cand = [doc.get(c) for c in "ABCD"]
        if all(c is not None for c in cand):
            opts = cand
    if q is None or opts is None:
        raise KeyError(f"Cannot find question/options in columns: {list(doc.keys())}")
    raw = _first(doc, ["target", "answer", "label", "correct_option", "correct_answer"])
    return {
        "id": f"{language}-{idx}",
        "language": language,
        "domain": str(_first(doc, ["domain", "Domain"]) or "NA"),
        "subject": str(_first(doc, ["subject", "Subject"]) or "NA"),
        "question": str(q).strip(),
        "options": [str(o).strip() for o in opts],
        "gold": _gold_index(raw, opts),
        "raw_target": raw,
    }


def build_samples(n_per_cell, seed, languages, out_path, n_shots=5):
    """Stratified sample: up to n_per_cell questions per (language x domain)."""
    from datasets import load_dataset
    rng = random.Random(seed)
    samples = []
    for lang in languages:
        test = load_dataset(DATASET, data_dir=lang, split="test")
        val = load_dataset(DATASET, data_dir=lang, split="validation")
        docs = [normalize(d, lang, i) for i, d in enumerate(test)]
        vdocs = [normalize(d, lang, f"val{i}") for i, d in enumerate(val)]
        docs = [d for d in docs if d["gold"] is not None]
        vdocs = [d for d in vdocs if d["gold"] is not None]
        cells = {}
        for d in docs:
            cells.setdefault(d["domain"], []).append(d)
        for dom in sorted(cells):
            pool = cells[dom]
            for d in rng.sample(pool, min(n_per_cell, len(pool))):
                same = [v for v in vdocs if v["subject"] == d["subject"]] or vdocs
                d = dict(d)
                d["shots"] = rng.sample(same, min(n_shots, len(same)))
                samples.append(d)
        print(f"[sample] {lang}: {sum(1 for s in samples if s['language']==lang)} questions")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(samples, f, ensure_ascii=False)
    return samples


def load_samples(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- prompts
def fmt_question(d, with_answer=False):
    s = f"Question: {d['question']}\n"
    for L, o in zip(LETTERS, d["options"]):
        s += f"{L}. {o}\n"
    s += "Answer:"
    if with_answer:
        s += f" {LETTERS[d['gold']]}\n\n"
    return s


def loglik_prompt(d, k):
    subj = d["subject"].replace("_", " ")
    head = f"The following are multiple choice questions (with answers) about {subj}.\n\n"
    shots = "".join(fmt_question(s, True) for s in d["shots"][:k])
    return head + shots + fmt_question(d)


def gen_prompt(d, cot):
    body = fmt_question(d)[:-len("Answer:")].rstrip()
    if cot:
        ins = ("Answer the following multiple choice question. Think step by step, but keep the reasoning "
               "brief (at most 6 short sentences) and write it IN ENGLISH even if the question is in another "
               "language. Then finish with a final line in exactly this format: 'Answer: X' where X is A, B, C or D.")
    else:
        ins = ("Answer the following multiple choice question. Respond with only the letter "
               "(A, B, C or D) of the correct option and nothing else.")
    return f"{ins}\n\n{body}\n"


# ---------------------------------------------------------------- parser
_EXPLICIT = re.compile(
    r"(?i:answer|correct option|correct choice|option)\s*(?:is|:|=|-|–)?\s*[:\-–]?\s*"
    r"\**\s*\(?\s*([A-D1-4])\s*\)?(?![A-Za-z])")
_LASTLINE = re.compile(r"^\W*([A-D])\W*$")
_FIRST = re.compile(r"^\W*([A-D])(?![A-Za-z])")


def _to_idx(tok):
    return LETTERS.index(tok) if tok in LETTERS else int(tok) - 1


def parse_answer(text, options=None):
    """Return (index or None, method). Strict: ambiguous output -> None."""
    t = (text or "").strip()
    if not t:
        return None, "empty"
    m = _EXPLICIT.findall(t)
    if m:
        return _to_idx(m[-1]), "explicit"
    lines = [l for l in t.splitlines() if l.strip()]
    if lines:
        m = _LASTLINE.match(lines[-1].strip())
        if m:
            return _to_idx(m.group(1)), "lastline"
    m = _FIRST.match(t)
    if m:
        return _to_idx(m.group(1)), "first"
    if options and lines:                       # last line equals exactly one option text
        hits = [i for i, o in enumerate(options) if o and o.strip() == lines[-1].strip().strip("*. ")]
        if len(hits) == 1:
            return hits[0], "optiontext"
    return None, "fail"
