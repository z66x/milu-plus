"""Run FIRST: checks GPU, HF access and the real MILU column names."""
import collections
import torch
print("CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB" if torch.cuda.is_available() else "")
try:
    import bitsandbytes; print("bitsandbytes", bitsandbytes.__version__)
except Exception as e:
    print("bitsandbytes problem:", e)
from datasets import load_dataset
from common import normalize
ds = load_dataset("ai4bharat/MILU", data_dir="Hindi", split="test")
print("\nColumns:", ds.column_names)
print("Raw row 0:", ds[0])
docs = [normalize(d, "Hindi", i) for i, d in enumerate(ds)]
print("\nNormalised:", {k: v for k, v in docs[0].items()})
print("raw target values:", collections.Counter(str(d["raw_target"]) for d in docs).most_common(8))
print("gold None:", sum(d["gold"] is None for d in docs), "of", len(docs))
print("gold dist:", collections.Counter(d["gold"] for d in docs))
print("domains:", collections.Counter(d["domain"] for d in docs))
