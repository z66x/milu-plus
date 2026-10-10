"""Aggregate runs/*.jsonl -> results/ tables + plots (incl. the Phase-2 slide table)."""
import glob, json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# EDIT if your group/paper uses a different split (assumption, not taken from the paper):
HIGH = {"English", "Hindi"}
CULTURE_KEYS = ("arts", "humanit", "law", "governance", "social")   # substring match on domain
STEM_KEYS = ("science", "engineering", "health", "environment")

rows = []
for p in glob.glob("runs/*.jsonl"):
    if "__smoke" in p and "--include-smoke" not in sys.argv:
        continue
    rows += [json.loads(l) for l in open(p)]
if not rows:
    sys.exit("no runs found in runs/")
df = pd.DataFrame(rows)
if "correct_fb" not in df: df["correct_fb"] = df["correct"]
df["correct_fb"] = df["correct_fb"].fillna(df["correct"])
df["tier"] = np.where(df.language.isin(HIGH), "high-res", "low-res")
dl = df.domain.str.lower()
df["group"] = np.select([dl.str.contains("|".join(CULTURE_KEYS)), dl.str.contains("|".join(STEM_KEYS))],
                        ["culture", "stem"], "other")
os.makedirs("results", exist_ok=True)
(df.assign(fail=df.pred.isna().astype(int)).groupby(["model", "mode", "language"]).fail.mean().mul(100).round(1)
   .unstack("language").to_csv("results/parse_fail_by_language.csv"))

def acc(g): return g.correct.mean() * 100
summ = df.groupby(["model", "mode"]).agg(n=("correct", "size"), acc=("correct", lambda s: s.mean() * 100),
        parse_fail=("pred", lambda s: s.isna().mean() * 100)).round(2)
summ.to_csv("results/summary.csv"); print(summ, "\n")
df.groupby(["model", "mode", "language"]).correct.mean().mul(100).round(1).unstack("language").to_csv("results/by_language.csv")
df.groupby(["model", "mode", "domain"]).correct.mean().mul(100).round(1).unstack("domain").to_csv("results/by_domain.csv")

# ---- slide-3 table: Model | Protocol | Avg | High-res | Low-res | STEM | Culture | n | parse-fail%
recs = []
for (m, mo), g in df.groupby(["model", "mode"]):
    f = lambda x: acc(x) if len(x) else np.nan
    recs.append({"model": m, "mode": mo, "Avg Acc": acc(g), "Acc+fallback": g.correct_fb.mean() * 100, "High-res": f(g[g.tier == "high-res"]),
                 "Low-res": f(g[g.tier == "low-res"]), "STEM": f(g[g.group == "stem"]),
                 "Culture": f(g[g.group == "culture"]), "n": len(g), "parse-fail %": g.pred.isna().mean() * 100})
t = pd.DataFrame(recs).set_index(["model", "mode"]).round(1)
t.to_csv("results/phase2_table.csv"); open("results/phase2_table.md", "w").write(t.to_markdown())
print(t.to_markdown(), "\n")

# ---- paired deltas (same question ids) with bootstrap CI
rng = np.random.default_rng(0); out = []
for m, g in df.groupby("model"):
    base = g[g["mode"].str.startswith("loglik")]
    if base.empty: continue
    base = base[base["mode"] == base["mode"].iloc[0]].set_index("id")
    for mode in [x for x in g["mode"].unique() if not x.startswith("loglik")]:
        o = g[g["mode"] == mode].set_index("id")
        ids = base.index.intersection(o.index)
        if not len(ids): continue
        d = (o.loc[ids, "correct"] - base.loc[ids, "correct"]).to_numpy() * 100
        bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        out.append({"model": m, "comparison": f"{mode} - {base['mode'].iloc[0]}", "n": len(ids),
                    "delta_pp": d.mean(), "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5)})
        for lvl in ("language", "domain"):
            lab = base.loc[ids, lvl]
            pd.Series(d, index=ids).groupby(lab).mean().round(2).to_csv(f"results/delta_{m.replace('/','_')}_{mode}_by_{lvl}.csv")
if out:
    D = pd.DataFrame(out).round(2); D.to_csv("results/deltas.csv"); print(D.to_markdown(index=False))

# ---- plots
ml = df.groupby(["language", "mode", "model"]).correct.mean().mul(100).reset_index()
for m, g in ml.groupby("model"):
    pv = g.pivot(index="language", columns="mode", values="correct")
    ax = pv.plot.bar(figsize=(9, 4), ylim=(0, 100)); ax.set_ylabel("accuracy %"); ax.set_title(m)
    plt.tight_layout(); plt.savefig(f"results/lang_{m.replace('/','_')}.png", dpi=150); plt.close()
print("\nwrote results/")
