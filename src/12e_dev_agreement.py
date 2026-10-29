"""
Step 12e - Development-set agreement, calibration and adjudication sheet
========================================================================

Input : outputs/12d_annotation/labels_dev_annotator1.csv, labels_dev_annotator2.csv, key.csv
        outputs/12d_semantic_episode_ranking.parquet, 12c_within_episode_pairs.parquet
        outputs/11_spec_blocks.parquet (texts for the adjudication sheet)

Reports (outputs/12e_dev_report.txt):
  0. Label hygiene      missing / invalid labels, label distribution per annotator
  1. Agreement          Cohen's kappa (7 labels, collapsed, binary COPIED) with bootstrap CI,
                        per-label specific agreement, confusion matrix, behaviour_differs kappa
                        agreement on short vs long items (a skimming check)
  2. Label x stratum    what each sampling stratum turned out to be (agreed items)
  3. Lexical axis       precision / recall of near & gray rules against human COPIED,
                        threshold grid, copies the lexical axis missed
  4. Semantic axis      SAME_BEHAVIOUR (and SAME_BEHAVIOUR or COPIED) by margin band,
                        weighted estimate of how many such pairs exist in the whole pool
  5. Lineage rule       what the lineage-only pairs turned out to be
Writes:
  outputs/12e_dev_items.csv                     merged labels + scores (analysis only)
  outputs/12d_annotation/adjudication_dev.html  disagreements, texts only (NO scores)
  outputs/12d_annotation/adjudication_dev.csv   fill final_label together

Usage:  python src\\12e_dev_agreement.py
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261019
LABELS = ["COPIED", "REVISION", "SAME_BEHAVIOUR", "RELATED", "BOILERPLATE", "UNRELATED", "UNCERTAIN"]
ALIASES = {"SAME_BEHAVIOR": "SAME_BEHAVIOUR", "SAMEBEHAVIOUR": "SAME_BEHAVIOUR", "SAME": "SAME_BEHAVIOUR",
           "COPY": "COPIED", "REVISED": "REVISION", "UNSURE": "UNCERTAIN", "RELATED_TOPIC": "RELATED"}
BD = ["no", "value", "entity", "other"]
COLLAPSE = {"COPIED": "COPIED", "REVISION": "REVISION", "SAME_BEHAVIOUR": "SAME_BEHAVIOUR",
            "RELATED": "OTHER", "BOILERPLATE": "OTHER", "UNRELATED": "OTHER", "UNCERTAIN": "UNCERTAIN"}
NEAR_J, NEAR_C, NEAR_C_MIN_SHARED, GRAY = 0.50, 0.80, 10, 0.30


def norm_label(x):
    if pd.isna(x) or not str(x).strip():
        return None
    s = str(x).strip().upper().replace(" ", "_").replace("-", "_")
    s = ALIASES.get(s, s)
    return s if s in LABELS else f"INVALID:{x}"


def norm_bd(x):
    if pd.isna(x) or not str(x).strip():
        return None
    s = str(x).strip().lower()
    return s if s in BD else f"INVALID:{x}"


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0:
        return np.nan
    cats = sorted(set(a) | set(b))
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return np.nan if pe == 1 else (po - pe) / (1 - pe)


def kappa_ci(a, b, n_boot=2000):
    rng = np.random.default_rng(SEED)
    a, b = np.asarray(a), np.asarray(b)
    ks = [kappa(a[i], b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(n_boot))]
    lo, hi = np.nanpercentile(ks, [2.5, 97.5])
    return kappa(a, b), lo, hi


def pr(pred, truth):
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    tp = (pred & truth).sum()
    p = tp / pred.sum() if pred.sum() else np.nan
    r = tp / truth.sum() if truth.sum() else np.nan
    return tp, int(pred.sum()), int(truth.sum()), p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--split", default="dev", choices=["dev", "heldout"])
    args = ap.parse_args()
    out = Path(args.out)
    pk = out / "12d_annotation"
    sp = args.split

    key = pd.read_csv(pk / "key.csv")
    key = key[key["split"] == sp]
    A = []
    for a in (1, 2):
        f = pd.read_csv(pk / f"labels_{sp}_annotator{a}.csv", dtype=str, encoding="utf-8-sig")
        f.columns = [c.strip().lower() for c in f.columns]
        f["item_id"] = f["item_id"].str.strip()
        A.append(f.rename(columns={"label": f"raw{a}", "behaviour_differs": f"rawbd{a}", "notes": f"notes{a}"})
                 [["item_id", f"raw{a}", f"rawbd{a}", f"notes{a}"]])
    D = key.merge(A[0], on="item_id", how="left").merge(A[1], on="item_id", how="left")
    for a in (1, 2):
        D[f"l{a}"] = D[f"raw{a}"].map(norm_label)
        D[f"bd{a}"] = D[f"rawbd{a}"].map(norm_bd)

    L = []
    w = L.append
    w(f"STEP 12e - {sp.upper()} SET: AGREEMENT AND CALIBRATION")
    w("=" * 64)
    w(f"Items in key: {len(D)}")

    # ---------------- 0. hygiene ---------------- #
    w("\n0. LABEL HYGIENE")
    bad = False
    for a in (1, 2):
        miss = D[f"l{a}"].isna().sum()
        inv = D[f"l{a}"].astype(str).str.startswith("INVALID").sum()
        bdinv = D[f"bd{a}"].astype(str).str.startswith("INVALID").sum()
        bdmiss = ((D[f"l{a}"] == "COPIED") & D[f"bd{a}"].isna()).sum()
        w(f"  annotator {a}: missing {miss}, invalid labels {inv}, invalid behaviour_differs {bdinv}, "
          f"COPIED without behaviour_differs {bdmiss}")
        if inv:
            w("    invalid values: " + ", ".join(sorted(set(D.loc[D[f'l{a}'].astype(str)
                                                               .str.startswith('INVALID'), f'raw{a}'].astype(str)))))
        bad |= bool(miss or inv)
    if bad:
        w("  -> Fix missing/invalid labels in the CSVs and rerun; affected items are dropped below.")
    dist = pd.DataFrame({f"annotator{a}": D[f"l{a}"].value_counts() for a in (1, 2)}).fillna(0).astype(int)
    w("  label distribution\n  " + dist.to_string().replace("\n", "\n  "))

    V = D[D["l1"].isin(LABELS) & D["l2"].isin(LABELS)].copy()
    if len(V) < 10:
        print("\n".join(L))
        raise SystemExit(f"\nOnly {len(V)} items have two valid labels - are these the filled-in label files? "
                         f"Check outputs/12d_annotation/labels_{sp}_annotator1.csv and ..._annotator2.csv.")
    V["agree"] = V["l1"] == V["l2"]
    V["final"] = np.where(V["agree"], V["l1"], "DISAGREE")

    # ---------------- 1. agreement ---------------- #
    w(f"\n1. AGREEMENT (n = {len(V)} items with two valid labels)")
    w(f"  raw agreement: {V['agree'].mean():.1%}")
    for name, f in (("7 labels", lambda s: s),
                    ("collapsed (COPIED/REVISION/SAME/OTHER/UNC)", lambda s: s.map(COLLAPSE)),
                    ("binary COPIED vs not", lambda s: s.eq("COPIED")),
                    ("binary reuse-or-same (COPIED|SAME_BEHAVIOUR) vs not",
                     lambda s: s.isin(["COPIED", "SAME_BEHAVIOUR"]))):
        k, lo, hi = kappa_ci(f(V["l1"]).to_numpy(), f(V["l2"]).to_numpy())
        w(f"  kappa {name:52s}: {k:.2f}  [95% CI {lo:.2f}, {hi:.2f}]")
    w("  (Landis-Koch: 0.41-0.60 moderate, 0.61-0.80 substantial, > 0.80 almost perfect)")
    w("\n  specific agreement per label (2*both / (n1 + n2))")
    for lab in LABELS:
        n1, n2 = (V["l1"] == lab).sum(), (V["l2"] == lab).sum()
        both = ((V["l1"] == lab) & (V["l2"] == lab)).sum()
        if n1 + n2:
            w(f"    {lab:15s} {2 * both / (n1 + n2):.2f}   (a1 {n1}, a2 {n2}, both {both})")
    cm = pd.crosstab(V["l1"].rename("annotator1"), V["l2"].rename("annotator2"))
    w("\n  confusion (rows annotator 1, columns annotator 2)\n  " + cm.to_string().replace("\n", "\n  "))
    C = V[(V["l1"] == "COPIED") & (V["l2"] == "COPIED") & V["bd1"].isin(BD) & V["bd2"].isin(BD)]
    if len(C):
        w(f"\n  behaviour_differs on items both called COPIED (n={len(C)}): "
          f"agreement {np.mean(C['bd1'] == C['bd2']):.1%}, kappa {kappa(C['bd1'], C['bd2']):.2f}")
        w("  " + pd.crosstab(C["bd1"].rename("a1"), C["bd2"].rename("a2")).to_string().replace("\n", "\n  "))

    texts = {}
    blocks_f = out / "11_spec_blocks.parquet"
    if blocks_f.exists():
        bl = pd.read_parquet(blocks_f)
        bl["uid"] = bl["episode_id"].astype(str) + "|" + bl["file_path"].astype(str) + "|" + bl["block_idx"].astype(str)
        col = "raw_text" if "raw_text" in bl.columns else "norm_text"
        texts = dict(zip(bl["uid"], bl[col].fillna("").astype(str)))
        V["len_tokens"] = [len(texts.get(a, "").split()) + len(texts.get(b, "").split())
                           for a, b in zip(V["uid_a"], V["uid_b"])]
        med = V["len_tokens"].median()
        s, l_ = V[V["len_tokens"] <= med], V[V["len_tokens"] > med]
        w(f"\n  skimming check (items split at median combined length {med:.0f} words):")
        w(f"    short items: agreement {s['agree'].mean():.1%}, kappa {kappa(s['l1'], s['l2']):.2f}  (n={len(s)})")
        w(f"    long  items: agreement {l_['agree'].mean():.1%}, kappa {kappa(l_['l1'], l_['l2']):.2f}  (n={len(l_)})")
        w("    A much lower kappa on long items suggests long items were not read in full.")

    # ---------------- 2. label x stratum ---------------- #
    w("\n2. LABEL x STRATUM (agreed label; DISAGREE = needs adjudication)")
    w("  " + pd.crosstab(V["stratum"], V["final"], margins=True).to_string().replace("\n", "\n  "))

    # ---------------- 3. lexical axis ---------------- #
    w("\n3. LEXICAL AXIS vs HUMAN COPIED (agreed items only; within the sample, not population rates)")
    G = V[V["agree"]].copy()
    truth = G["final"].eq("COPIED")
    truth_cr = G["final"].isin(["COPIED", "REVISION"])
    near = (G["j5"] >= NEAR_J) | ((G["c5"] >= NEAR_C) & (G["shared5"] >= NEAR_C_MIN_SHARED)) | (G["j5"] >= 0.999)
    gray = np.maximum(G["j5"], G["c5"]) >= GRAY
    for nm, pred in (("near rule", near), ("near or gray (>= 0.30)", gray)):
        for tn, t in (("COPIED", truth), ("COPIED|REVISION", truth_cr)):
            tp, npred, ntrue, p, r = pr(pred, t)
            w(f"  {nm:24s} vs {tn:16s}: precision {p:.2f} ({tp}/{npred}), recall {r:.2f} ({tp}/{ntrue})")
    w("\n  threshold grid on max(J5, C5) vs human COPIED:")
    sc = np.maximum(G["j5"], G["c5"])
    for t in (0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80):
        tp, npred, ntrue, p, r = pr(sc >= t, truth)
        w(f"    >= {t:.2f}: precision {p:.2f}  recall {r:.2f}  (flagged {npred}, true {ntrue})")
    miss = G[truth & ~gray]
    w(f"\n  human COPIED below the gray threshold (missed by the lexical axis): {len(miss)}")
    if len(miss):
        w("  by stratum: " + ", ".join(f"{k}={v}" for k, v in miss["stratum"].value_counts().items()))
        bdm = miss["bd1"].where(miss["bd1"] == miss["bd2"], "disagree").value_counts()
        w("  behaviour_differs among them: " + ", ".join(f"{k}={v}" for k, v in bdm.items()))

    # ---------------- 4. semantic axis ---------------- #
    w("\n4. SEMANTIC AXIS (sem_top / sem_mid / sem_tail items)")
    S = V[V["stratum"].isin(["sem_top", "sem_mid", "sem_tail"])].copy()
    if len(S):
        bands = [(1.06, 9), (1.04, 1.06), (1.02, 1.04), (0, 1.02)]
        w("  margin band     n  agreed  SAME   COPIED  REVISION  SAME|COPIED  (shares of agreed)")
        for lo, hi in bands:
            b = S[(S["margin"] >= lo) & (S["margin"] < hi)]
            g = b[b["agree"]]
            def sh(lbls):
                return f"{g['final'].isin(lbls).mean():.2f}" if len(g) else "  - "
            w(f"  [{lo:.2f},{hi if hi < 9 else 'max':>5}) {len(b):4d} {len(g):6d}   {sh(['SAME_BEHAVIOUR']):>4}   "
              f"{sh(['COPIED']):>5}   {sh(['REVISION']):>6}   {sh(['SAME_BEHAVIOUR', 'COPIED']):>8}")
        rk = out / "12d_semantic_episode_ranking.parquet"
        if rk.exists():
            n_rank = len(pd.read_parquet(rk, columns=["sem_rank"]))
            Npop = {"sem_top": min(60, n_rank), "sem_mid": max(0, min(600, n_rank) - 60),
                    "sem_tail": max(0, n_rank - 600)}
            w(f"\n  weighted population estimates (ranked pool = {n_rank:,} episode pairs; dev share of each")
            w("  stratum treated as a random sample of it):")
            for lbls, nm in ((["SAME_BEHAVIOUR"], "SAME_BEHAVIOUR"), (["COPIED"], "COPIED (lexically missed)"),
                             (["SAME_BEHAVIOUR", "COPIED"], "SAME|COPIED")):
                tot = 0.0
                parts = []
                for st, N in Npop.items():
                    g = S[(S["stratum"] == st) & S["agree"]]
                    if len(g):
                        est = g["final"].isin(lbls).mean() * N
                        tot += est
                        parts.append(f"{st} {est:.0f}")
                w(f"    {nm:28s}: ~{tot:.0f}   ({', '.join(parts)})")
    R = V[(V["stratum"] == "random_ctrl") & V["agree"]]
    if len(R):
        w(f"\n  random non-reuse controls (n agreed = {len(R)}): " +
          ", ".join(f"{k}={v}" for k, v in R["final"].value_counts().items()))
        wf = out / "12c_within_episode_pairs.parquet"
        if wf.exists():
            wp = pd.read_parquet(wf)
            pool = wp[wp["shared_language"] & ~wp["has_lineage_match"] &
                      wp["category"].isin(["reworded_candidate", "related_candidate", "unrelated"])]
            hit = R["final"].isin(["COPIED", "SAME_BEHAVIOUR"]).mean()
            w(f"  -> share COPIED|SAME among random controls {hit:.1%}; pool {len(pool):,} episode pairs;"
              f" implied unretrieved reuse ~{hit * len(pool):,.0f} pairs (very imprecise at this n)")

    # ---------------- 5. lineage ---------------- #
    Lg = V[V["stratum"] == "lineage"]
    w(f"\n5. LINEAGE RULE CHECK (n = {len(Lg)} lineage-only pairs in this split)")
    if len(Lg):
        w("  " + ", ".join(f"{k}={v}" for k, v in Lg["final"].value_counts().items()))
        w("  Expect mostly REVISION (or COPIED with behaviour_differs=no). Many SAME/RELATED would mean the")
        w("  lineage rule also removes different requirements that merely share a capability.")

    # ---------------- outputs ---------------- #
    D.merge(V[["item_id", "agree", "final"]], on="item_id", how="left").to_csv(out / f"12e_{sp}_items.csv", index=False)
    dis = V[~V["agree"]].sort_values("item_id")
    adj = pd.DataFrame({"item_id": dis["item_id"], "annotator1": dis["l1"], "annotator2": dis["l2"],
                        "bd1": dis["bd1"], "bd2": dis["bd2"], "final_label": "", "final_behaviour_differs": "",
                        "reason": ""})
    adj.to_csv(pk / f"adjudication_{sp}.csv", index=False)
    parts = [f"<!doctype html><meta charset='utf-8'><title>Adjudication - {sp}</title>",
             "<style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:2em auto;padding:0 1em}"
             ".pair{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:0 0 2.5em}"
             ".pair pre{white-space:pre-wrap;background:#f6f6f6;padding:10px;margin:0;font-size:13px}"
             ".lab{background:#fff4d6;padding:6px 10px;margin:.3em 0}</style>",
             f"<h1>Adjudication - {sp} ({len(dis)} disagreements)</h1>",
             "<p>Discuss each item together, agree a final label using the codebook, write it in "
             f"adjudication_{sp}.csv with a one-line reason. If an item shows the codebook is unclear, "
             "note the rule you agreed - it becomes a codebook change before the held-out packet.</p>"]
    for _, x in dis.iterrows():
        n1 = "" if pd.isna(x["notes1"]) else f" - <i>{html.escape(str(x['notes1']))}</i>"
        n2 = "" if pd.isna(x["notes2"]) else f" - <i>{html.escape(str(x['notes2']))}</i>"
        parts.append(f"<h3>{x['item_id']}</h3><div class='lab'>annotator 1: <b>{x['l1']}</b> "
                     f"{'' if pd.isna(x['bd1']) else '(' + str(x['bd1']) + ')'}{n1}<br>annotator 2: <b>{x['l2']}</b> "
                     f"{'' if pd.isna(x['bd2']) else '(' + str(x['bd2']) + ')'}{n2}</div><div class='pair'>"
                     f"<div><pre>{html.escape(texts.get(x['uid_a'], '(text not found)'))}</pre></div>"
                     f"<div><pre>{html.escape(texts.get(x['uid_b'], '(text not found)'))}</pre></div></div>")
    (pk / f"adjudication_{sp}.html").write_text("\n".join(parts), encoding="utf-8")
    w(f"\nWrote {len(dis)} disagreements to {pk / f'adjudication_{sp}.html'} and adjudication_{sp}.csv")
    s = "\n".join(L)
    (out / f"12e_{sp}_report.txt").write_text(s, encoding="utf-8")
    print(s)


if __name__ == "__main__":
    main()