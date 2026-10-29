"""
Step 12d - Repo-relative semantic scoring + blinded annotation packets
=====================================================================

Why: the Step 12c semantic bins used a CROSS-repo null and took the max over all block
pairs of an episode pair. Within one repository, requirements share vocabulary and topic,
so 13% of within-repo episode pairs became "reworded candidates" - an artifact. 12c's
LEXICAL results stand; its semantic bins are superseded by this step.

Semantic score (no new embedding - reuses the 12c cache):
  * Neighbourhood: for each block, the other blocks of the SAME repo, in other episodes,
    excluding lineage (same capability / feature / change id).
  * Ratio margin (Artetxe & Schwenk 2019, bitext mining):
        margin(i,j) = cos(i,j) / ( (mean top-k cos of i + mean top-k cos of j) / 2 ),  k = 4
    i.e. "how much more similar are these two than each one is to its usual close
    neighbours in this repo". Removes repo topic and hub blocks.
  * Mutual nearest neighbours: j in top-3 of i AND i in top-3 of j.
  * Only lexically LOW pairs (not exact/near/gray in 12c) enter the semantic ranking.
  * Episode-pair score = best margin among its mutual, lexically-low, non-lineage block
    pairs. Episode pairs are RANKED; no threshold is set before human labels exist.

Annotation sample (one item = the representative block pair of one episode pair):
  copied       all within-repo copied pairs (shared language, no lineage match)
  gray         all within-repo gray pairs
  sem_top      semantic ranks 1-60
  sem_mid      random 50 from ranks 61-600
  sem_tail     random 30 from ranks 601+
  random_ctrl  random 40 non-reuse within-repo pairs (most similar block pair shown)
  lineage      random 30 lineage-only pairs (checks the lineage rule)
  cross        cross-repo copied pairs (up to 40) + random 30 cross-repo gray pairs
Split by repository into DEVELOPMENT (~40%) and HELD-OUT. Items shuffled, A/B order
randomized, annotators see only the two texts and the codebook.

Outputs (outputs/):
  12d_semantic_block_pairs.parquet    mutual-NN within-repo block pairs with margin
  12d_semantic_episode_ranking.parquet ranked candidate episode pairs
  12d_semantic_top40.txt              top 40 pairs, text side by side (READ THIS)
  12d_summary.txt
  12d_annotation/packet_{dev,heldout}.html
  12d_annotation/labels_{dev,heldout}_annotator{1,2}.csv
  12d_annotation/key.csv              strata + scores. Do NOT give to annotators.

Usage:  python src\\12d_semantic_rescore_and_packets.py
        (add --emb-key KEY if more than one complete embedding set is cached)
"""

from __future__ import annotations

import argparse
import html
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261019
K_MARGIN, K_MUTUAL = 4, 3
EMB_CHUNK = 2000            # must match 12c
ROW_CHUNK = 1500
DEV_SHARE = 0.40
N_SEM_TOP, N_SEM_MID, N_SEM_TAIL, SEM_MID_END = 60, 50, 30, 600
N_RANDOM, N_LINEAGE, N_CROSS_GRAY, N_CROSS_COPIED = 40, 30, 30, 40
NEAR_J, NEAR_C, NEAR_C_MIN_SHARED = 0.50, 0.80, 10

CJK_CLASS = r"぀-ヿ㐀-䶿一-鿿가-힯豈-﫿"
TOKEN_RE = re.compile(rf"[{CJK_CLASS}]|[^\s{CJK_CLASS}]+")

CODEBOOK = """
<h2>Codebook - requirement pair labels</h2>
<p>Each item shows two requirement blocks (A and B) from two different pull requests.
Judge <b>only the requirement text</b>. Do not look up the projects, do not guess where the
pair came from, and do not discuss items with the other annotator until both label files
are final. Ignore template scaffolding (headings, "Given/When/Then" keywords, "MUST/SHALL",
requirement numbers such as FR-012).</p>
<table>
<tr><th>Label</th><th>Use when</th></tr>
<tr><td><b>COPIED</b></td><td>One was written by copying the other: the same sentences or the same
sentence structure reused, possibly with changed values, a few clauses added or removed, or
the subject swapped for another entity (e.g. <i>scan_vcf_zarr / read_vcf_zarr</i> rewritten as
<i>scan_bigbed / read_bigbed</i>). There must be shared <i>specific</i> wording - phrases, enumerations,
examples, clause order - not just the usual sentence form ("System MUST support filtering by X" vs
"... by Y" alone is not COPIED). Fill <b>behaviour_differs</b>.</td></tr>
<tr><td><b>REVISION</b></td><td>Both state the SAME requirement for the SAME feature, and one reads as a
later version or a restatement of the other ("MUST continue to...", "MUST remain...",
"preserve the existing..."), not as reuse for a new feature.</td></tr>
<tr><td><b>SAME_BEHAVIOUR</b></td><td>Worded independently (no copied sentence structure), but both
require the same system behaviour: an implementation that satisfies one would satisfy the other.</td></tr>
<tr><td><b>RELATED</b></td><td>Same feature area or topic, but they require different behaviour.</td></tr>
<tr><td><b>BOILERPLATE</b></td><td>The only shared content is generic: template phrases, generic lists
of questions, generic quality rules ("styling MUST be theme-safe", "errors are logged"), or a
<b>project-wide convention</b> restated for each new item (e.g. "the resource SHALL be registered in
provider.go", "unit tests SHALL use gomonkey and run with go test -gcflags=all=-l") - text that any
requirement of that kind in the project would contain. Write <i>convention</i> in notes for these.</td></tr>
<tr><td><b>UNRELATED</b></td><td>Different topics.</td></tr>
<tr><td><b>UNCERTAIN</b></td><td>Use sparingly, and say why in <i>notes</i>.</td></tr>
</table>
<p><b>behaviour_differs</b> (fill only for COPIED), one of:<br>
<i>no</i> - only the wording changed;<br>
<i>value</i> - a value, limit, condition, actor or negation changed;<br>
<i>entity</i> - the same requirement applied to a different entity, format, component or command;<br>
<i>other</i> - explain in notes.</p>
<p>If both REVISION and COPIED seem possible, choose REVISION only when the text itself signals the
same feature (e.g. "continue", "remain", "existing"); otherwise COPIED.</p>
<p>Write labels in your own CSV file using exactly these words, one row per item id.</p>
"""


def log(msg, t0):
    print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)


def shingles(text, n=5):
    toks = TOKEN_RE.findall(text or "")
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[k:k + n]) for k in range(len(toks) - n + 1)}


def lex_pair(ta, tb):
    a, b = shingles(ta), shingles(tb)
    if not a or not b:
        return 0.0, 0.0, 0
    inter = len(a & b)
    return inter / len(a | b), inter / min(len(a), len(b)), inter


def load_embeddings(cache_dir: Path, n: int, key: str | None):
    by_key = defaultdict(list)
    for f in cache_dir.glob("*.npy"):
        k, _, s = f.stem.rpartition("_")
        if s.isdigit():
            by_key[k].append((int(s), f))
    expected = list(range(0, n, EMB_CHUNK))
    complete = {k: sorted(v) for k, v in by_key.items() if [s for s, _ in sorted(v)] == expected}
    if key is None:
        if len(complete) != 1:
            info = {k: len(v) for k, v in by_key.items()}
            raise SystemExit(f"need exactly one complete embedding set for {n} blocks; "
                             f"found complete={list(complete)} all={info}. Pass --emb-key.")
        key = next(iter(complete))
    if key not in complete:
        raise SystemExit(f"embedding key {key} is incomplete or missing")
    E = np.vstack([np.load(f) for _, f in complete[key]]).astype(np.float32)
    if E.shape[0] != n:
        raise SystemExit(f"embedding rows {E.shape[0]} != blocks {n}")
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    return E, key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--cache", default="cache/12c")
    ap.add_argument("--variant", default="norm_text")
    ap.add_argument("--show-col", default="raw_text", help="text shown to annotators")
    ap.add_argument("--emb-key", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    pk = out / "12d_annotation"
    pk.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # ---------------- load (same block set and order as 12c) ---------------- #
    blocks = pd.read_parquet(out / "11_spec_blocks.parquet")
    B = blocks[blocks["primary"] & blocks["substantive"] & ~blocks["dup_within_episode"]].reset_index(drop=True)
    B["uid"] = B["episode_id"].astype(str) + "|" + B["file_path"].astype(str) + "|" + B["block_idx"].astype(str)
    idx12c = pd.read_parquet(out / "12c_block_index.parquet")
    if len(idx12c) != len(B) or not (idx12c["uid"].to_numpy() == B["uid"].to_numpy()).all():
        raise SystemExit("block order differs from 12c_block_index - rerun 12c on the same Step 11 output")
    n = len(B)
    show_col = args.show_col if args.show_col in B.columns else args.variant
    E, emb_key = load_embeddings(Path(args.cache) / "emb", n, args.emb_key)
    log(f"{n:,} blocks, embeddings {E.shape} (key {emb_key})", t0)

    ep_codes, ep_names = pd.factorize(B["episode_id"])
    repo_codes, repo_names = pd.factorize(B["repo"])
    lineage_cols = [c for c in ("capability", "feature_key", "change_id") if c in B.columns]
    lin_codes = [pd.factorize(B[c].replace("", np.nan))[0] for c in lineage_cols]
    rows_of_ep = pd.Series(np.arange(n)).groupby(B["episode_id"]).apply(lambda s: s.to_numpy()).to_dict()

    wbp = pd.read_parquet(out / "12c_within_block_pairs.parquet")
    nonlow = wbp[wbp["lex_bin"] <= 2]
    nonlow_keys = set((np.minimum(nonlow["i"], nonlow["j"]).astype(np.int64) * n +
                       np.maximum(nonlow["i"], nonlow["j"]).astype(np.int64)).tolist())
    wep = pd.read_parquet(out / "12c_within_episode_pairs.parquet")
    log(f"12c: {len(wbp):,} stored within-repo block pairs, {len(wep):,} episode pairs", t0)

    # ---------------- repo-relative margin + mutual NN ---------------- #
    sem_rows = []
    groups = pd.Series(np.arange(n)).groupby(repo_codes).apply(lambda s: s.to_numpy())
    for gi, g in enumerate(groups.index):
        idx = groups[g]
        m = len(idx)
        if m < 2 or len(np.unique(ep_codes[idx])) < 2:
            continue
        Er, epr = E[idx], ep_codes[idx]
        linr = [lc[idx] for lc in lin_codes]
        r = np.full(m, np.nan)
        top = np.full((m, K_MUTUAL), -1, dtype=np.int64)
        for s in range(0, m, ROW_CHUNK):
            a = np.arange(s, min(s + ROW_CHUNK, m))
            S = Er[a] @ Er.T
            bad = epr[a][:, None] == epr[None, :]
            for lc in linr:
                bad |= (lc[a][:, None] == lc[None, :]) & (lc[a][:, None] >= 0)
            S[bad] = -np.inf
            kk = min(max(K_MARGIN, K_MUTUAL), m - 1)
            part = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            vals = np.take_along_axis(S, part, axis=1)
            order = np.argsort(-vals, axis=1)
            part, vals = np.take_along_axis(part, order, axis=1), np.take_along_axis(vals, order, axis=1)
            fin = np.isfinite(vals)
            vk = np.where(fin[:, :K_MARGIN], vals[:, :K_MARGIN], np.nan)
            with np.errstate(invalid="ignore"):
                r[a] = np.nanmean(vk, axis=1) if vk.size else np.nan
            km = min(K_MUTUAL, part.shape[1])
            top[a, :km] = np.where(fin[:, :km], part[:, :km], -1)
        topsets = [set(t[t >= 0].tolist()) for t in top]
        for i in range(m):
            for j in topsets[i]:
                if j > i and i in topsets[j]:
                    gi_, gj_ = int(idx[i]), int(idx[j])
                    c = float(Er[i] @ Er[j])
                    sem_rows.append((gi_, gj_, c, c / ((r[i] + r[j]) / 2),
                                     (gi_ * n + gj_) not in nonlow_keys))
        if gi % 50 == 0:
            log(f"semantic: {gi + 1}/{len(groups)} repos", t0)

    sp = pd.DataFrame(sem_rows, columns=["i", "j", "cos", "margin", "lex_low"])
    sp["episode_i"] = ep_names[ep_codes[sp["i"]]]
    sp["episode_j"] = ep_names[ep_codes[sp["j"]]]
    sw = sp["episode_i"] > sp["episode_j"]
    sp.loc[sw, ["episode_i", "episode_j"]] = sp.loc[sw, ["episode_j", "episode_i"]].to_numpy()
    sp.to_parquet(out / "12d_semantic_block_pairs.parquet", index=False)
    log(f"mutual top-{K_MUTUAL} block pairs: {len(sp):,} ({int(sp['lex_low'].sum()):,} lexically low)", t0)

    # ---------------- episode ranking ---------------- #
    w = wep.copy()
    sw = w["episode_i"] > w["episode_j"]
    w.loc[sw, ["episode_i", "episode_j"]] = w.loc[sw, ["episode_j", "episode_i"]].to_numpy()
    base = w[w["shared_language"] & ~w["has_lineage_match"]]
    pool_cats = {"reworded_candidate", "related_candidate", "unrelated"}
    pool = base[base["category"].isin(pool_cats)]
    best = (sp[sp["lex_low"]].sort_values("margin", ascending=False)
            .drop_duplicates(["episode_i", "episode_j"]))
    rank = pool.merge(best[["episode_i", "episode_j", "i", "j", "cos", "margin"]], on=["episode_i", "episode_j"])
    rank = rank.sort_values("margin", ascending=False).reset_index(drop=True)
    rank["sem_rank"] = np.arange(1, len(rank) + 1)
    rank.to_parquet(out / "12d_semantic_episode_ranking.parquet", index=False)

    text = B[show_col].fillna("").astype(str).to_numpy()
    ltext = B[args.variant].fillna("").astype(str).to_numpy()
    with open(out / "12d_semantic_top40.txt", "w", encoding="utf-8") as f:
        for _, x in rank.head(40).iterrows():
            f.write(f"#{x.sem_rank}  margin {x.margin:.3f}  cos {x.cos:.3f}  repo {x.repo_i}\n")
            f.write("  A: " + " ".join(text[int(x.i)].split())[:400] + "\n")
            f.write("  B: " + " ".join(text[int(x.j)].split())[:400] + "\n\n")

    # ---------------- annotation sample ---------------- #
    lexpairs = wbp[(wbp["lex_bin"] <= 2) & ~wbp["lineage"]].copy()
    lexpairs["episode_i"] = ep_names[ep_codes[lexpairs["i"]]]
    lexpairs["episode_j"] = ep_names[ep_codes[lexpairs["j"]]]
    sw = lexpairs["episode_i"] > lexpairs["episode_j"]
    lexpairs.loc[sw, ["episode_i", "episode_j"]] = lexpairs.loc[sw, ["episode_j", "episode_i"]].to_numpy()
    lexpairs["score"] = np.maximum(lexpairs["j5"], lexpairs["c5"]) + (lexpairs["lex_bin"] == 0)
    rep_lex = (lexpairs.sort_values("score", ascending=False)
               .drop_duplicates(["episode_i", "episode_j"]).set_index(["episode_i", "episode_j"]))

    def best_cos_pair(e1, e2, allow_lineage):
        a, b = rows_of_ep[e1], rows_of_ep[e2]
        S = E[a] @ E[b].T
        if not allow_lineage:
            for lc in lin_codes:
                S[(lc[a][:, None] == lc[b][None, :]) & (lc[a][:, None] >= 0)] = -np.inf
        k = int(np.argmax(S))
        if not np.isfinite(S.flat[k]):
            return None
        return int(a[k // len(b)]), int(b[k % len(b)])

    items = []

    def add(stratum, e1, e2, i, j, extra=None):
        items.append(dict(stratum=stratum, episode_a=e1, episode_b=e2, i=i, j=j,
                          repo=B["repo"].iat[i], repo_b=B["repo"].iat[j], **(extra or {})))

    for cat in ("copied", "gray"):
        for _, x in base[base["category"] == cat].iterrows():
            key = (x.episode_i, x.episode_j)
            if key in rep_lex.index:
                p = rep_lex.loc[key]
                add(cat, *key, int(p["i"]), int(p["j"]))

    def take(df, k):
        return df.sample(n=min(k, len(df)), random_state=int(rng.integers(1 << 31))) if len(df) else df

    sem_parts = [("sem_top", rank.head(N_SEM_TOP)),
                 ("sem_mid", take(rank.iloc[N_SEM_TOP:SEM_MID_END], N_SEM_MID)),
                 ("sem_tail", take(rank.iloc[SEM_MID_END:], N_SEM_TAIL))]
    for st, df in sem_parts:
        for _, x in df.iterrows():
            add(st, x.episode_i, x.episode_j, int(x.i), int(x.j),
                {"margin": x.margin, "sem_rank": int(x.sem_rank)})

    used = {(it["episode_a"], it["episode_b"]) for it in items}
    ctrl = pool[[(a, b) not in used for a, b in zip(pool["episode_i"], pool["episode_j"])]]
    for _, x in take(ctrl, N_RANDOM).iterrows():
        p = best_cos_pair(x.episode_i, x.episode_j, allow_lineage=False)
        if p:
            add("random_ctrl", x.episode_i, x.episode_j, *p)
    lin_only = w[(w["category"] == "lineage_only") & w["shared_language"]]
    for _, x in take(lin_only, N_LINEAGE).iterrows():
        p = best_cos_pair(x.episode_i, x.episode_j, allow_lineage=True)
        if p:
            add("lineage", x.episode_i, x.episode_j, *p)

    cbp = pd.read_parquet(out / "12c_cross_block_pairs.parquet")
    cep = pd.read_parquet(out / "12c_cross_episode_pairs.parquet")
    if len(cbp) and len(cep):
        cb = cbp[(cbp["provenance"] != "shared_ancestry") & (cbp["lex_bin"] <= 2)].copy()
        cb["episode_i"] = ep_names[ep_codes[cb["i"]]]
        cb["episode_j"] = ep_names[ep_codes[cb["j"]]]
        sw = cb["episode_i"] > cb["episode_j"]
        cb.loc[sw, ["episode_i", "episode_j"]] = cb.loc[sw, ["episode_j", "episode_i"]].to_numpy()
        cb["score"] = np.maximum(cb["j5"], cb["c5"]) + (cb["lex_bin"] == 0)
        rep_c = (cb.sort_values("score", ascending=False)
                 .drop_duplicates(["episode_i", "episode_j"]).set_index(["episode_i", "episode_j"]))
        c2 = cep.copy()
        sw = c2["episode_i"] > c2["episode_j"]
        c2.loc[sw, ["episode_i", "episode_j"]] = c2.loc[sw, ["episode_j", "episode_i"]].to_numpy()
        for st, df in (("cross_copied", take(c2[c2["category"] == "copied"], N_CROSS_COPIED)),
                       ("cross_gray", take(c2[c2["category"] == "gray"], N_CROSS_GRAY))):
            for _, x in df.iterrows():
                key = (x.episode_i, x.episode_j)
                if key in rep_c.index:
                    p = rep_c.loc[key]
                    add(st, *key, int(p["i"]), int(p["j"]))

    K = pd.DataFrame(items)
    lex = [lex_pair(ltext[i], ltext[j]) for i, j in zip(K["i"], K["j"])]
    K["j5"], K["c5"], K["shared5"] = zip(*lex)
    K["cos"] = [float(E[i] @ E[j]) for i, j in zip(K["i"], K["j"])]
    K["uid_a"], K["uid_b"] = B["uid"].to_numpy()[K["i"]], B["uid"].to_numpy()[K["j"]]

    # ---------------- split by repository, shuffle, write packets ---------------- #
    repos = K["repo"].drop_duplicates().to_numpy().copy()
    rng.shuffle(repos)
    cnt = K["repo"].value_counts()
    dev, acc = set(), 0
    for rp in repos:
        if acc < DEV_SHARE * len(K):
            dev.add(rp)
            acc += cnt[rp]
    K["split"] = np.where(K["repo"].isin(dev), "dev", "heldout")
    K = K.sample(frac=1, random_state=SEED).reset_index(drop=True)
    flip = rng.random(len(K)) < 0.5
    K["shown_a"] = np.where(flip, K["j"], K["i"])
    K["shown_b"] = np.where(flip, K["i"], K["j"])
    for sp_name, prefix in (("dev", "D"), ("heldout", "H")):
        S = K[K["split"] == sp_name].reset_index()
        K.loc[S["index"], "item_id"] = [f"{prefix}-{k + 1:03d}" for k in range(len(S))]
    for sp_name in ("dev", "heldout"):
        S = K[K["split"] == sp_name]
        parts = [f"<!doctype html><meta charset='utf-8'><title>Spec pairs - {sp_name}</title>",
                 "<style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:2em auto;padding:0 1em}"
                 "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px;vertical-align:top}"
                 ".pair{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:0 0 2.5em}"
                 ".pair pre{white-space:pre-wrap;background:#f6f6f6;padding:10px;margin:0;font-size:13px}"
                 "h3{margin:1.5em 0 .4em}</style>", CODEBOOK]
        for _, x in S.iterrows():
            parts.append(f"<h3>{x.item_id}</h3><div class='pair'>"
                         f"<div><b>A</b><pre>{html.escape(text[int(x.shown_a)])}</pre></div>"
                         f"<div><b>B</b><pre>{html.escape(text[int(x.shown_b)])}</pre></div></div>")
        (pk / f"packet_{sp_name}.html").write_text("\n".join(parts), encoding="utf-8")
        for a in (1, 2):
            lf = pk / f"labels_{sp_name}_annotator{a}.csv"
            if lf.exists() and pd.read_csv(lf, dtype=str)["label"].notna().any():
                print(f"[keep] {lf.name} already has labels - not overwritten")
                continue
            pd.DataFrame({"item_id": S["item_id"], "label": "", "behaviour_differs": "", "notes": ""}) \
                .sort_values("item_id").to_csv(lf, index=False)
    keep = ["item_id", "split", "stratum", "repo", "repo_b", "episode_a", "episode_b", "uid_a", "uid_b",
            "j5", "c5", "shared5", "cos", "margin", "sem_rank"]
    K[[c for c in keep if c in K.columns]].sort_values("item_id").to_csv(pk / "key.csv", index=False)

    # ---------------- summary ---------------- #
    L = ["STEP 12d - REPO-RELATIVE SEMANTIC SCORE + ANNOTATION PACKETS", "=" * 64,
         f"Embedding cache key: {emb_key} | margin k={K_MARGIN}, mutual top-{K_MUTUAL}", "",
         f"Mutual-NN within-repo block pairs: {len(sp):,}  (lexically low: {int(sp['lex_low'].sum()):,})",
         f"Candidate pool (shared language, no lineage, not copied/gray): {len(pool):,} episode pairs",
         f"  ... with >= 1 mutual, lexically-low block pair (ranked): {len(rank):,}", ""]
    if len(rank):
        q = rank["margin"].quantile([.5, .9, .99]).round(3).to_dict()
        L.append(f"Episode-pair margin: median {q[.5]}, p90 {q[.9]}, p99 {q[.99]}, max {rank['margin'].max():.3f}")
        L.append("Ranked pairs by margin (counts / distinct repos / largest repo share):")
        for t in (1.02, 1.04, 1.06, 1.08, 1.10, 1.15):
            s = rank[rank["margin"] >= t]
            sh = 100 * s["repo_i"].value_counts().iloc[0] / len(s) if len(s) else 0
            L.append(f"  margin >= {t:.2f}: {len(s):6d} pairs, {s['repo_i'].nunique():4d} repos, top repo {sh:4.0f}%")
        bias = rank.head(100)["n_block_pairs"].median() / max(rank["n_block_pairs"].median(), 1)
        L.append(f"Size-bias check: median block pairs in top 100 / in all ranked = {bias:.2f} "
                 "(>> 1 means big episodes dominate the top)")
        L.append("These thresholds are DESCRIPTIVE ONLY. Which margin means SAME_BEHAVIOUR is")
        L.append("decided by the sem_top / sem_mid / sem_tail labels on the dev packet.")
    L += ["", "ANNOTATION SAMPLE (stratum x split)",
          pd.crosstab(K["stratum"], K["split"], margins=True).to_string(), "",
          f"Packets in {pk}. Give annotators ONLY packet_*.html and their own labels_*.csv.",
          "Keep key.csv away from annotators until both label files are final.",
          f"Runtime {time.time() - t0:.0f}s"]
    s = "\n".join(L)
    (out / "12d_summary.txt").write_text(s, encoding="utf-8")
    print("\n" + s)


if __name__ == "__main__":
    main()