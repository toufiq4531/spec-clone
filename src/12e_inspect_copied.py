"""Print the agreed-COPIED dev items with the LOWEST lexical overlap, texts side by side.
Usage: python src\\12e_inspect_copied.py  [--n 15]   -> also writes outputs\\12e_copied_lowest_overlap.txt"""
import argparse
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="outputs")
ap.add_argument("--n", type=int, default=15)
args = ap.parse_args()
out = Path(args.out)
d = pd.read_csv(out / "12e_dev_items.csv")
d = d[d["final"] == "COPIED"].copy()
d["overlap"] = d[["j5", "c5"]].max(axis=1)
d = d.sort_values("overlap").head(args.n)
b = pd.read_parquet(out / "11_spec_blocks.parquet")
b["uid"] = b["episode_id"].astype(str) + "|" + b["file_path"].astype(str) + "|" + b["block_idx"].astype(str)
col = "raw_text" if "raw_text" in b.columns else "norm_text"
t = dict(zip(b["uid"], b[col].fillna("").astype(str)))
lines = []
for _, x in d.iterrows():
    lines.append(f"{x['item_id']}  stratum={x['stratum']}  max(J5,C5)={x['overlap']:.2f}  "
                 f"behaviour_differs={x['bd1']}/{x['bd2']}")
    lines.append("  A: " + " ".join(t.get(x["uid_a"], "?").split())[:500])
    lines.append("  B: " + " ".join(t.get(x["uid_b"], "?").split())[:500] + "\n")
s = "\n".join(lines)
(out / "12e_copied_lowest_overlap.txt").write_text(s, encoding="utf-8")
print(s)