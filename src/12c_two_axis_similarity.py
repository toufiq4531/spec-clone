"""
Step 12c - Exhaustive two-axis specification similarity (lexical x semantic)
=============================================================================

Replaces the MinHash/LSH retrieval of Step 12 (Step 12 outputs are kept unchanged).
See docs/deviations.md, entry 2026-10-19.

Input : outputs/11_spec_blocks.parquet   (primary, substantive, de-duplicated blocks)
        outputs/10b_episode_manifest.csv  (episode languages, tool)
        outputs/12_repo_families.csv      (shared-root-commit families from Step 12)

Axis 1 - lexical (exhaustive, no candidate retrieval):
    Word 5-shingles (primary) and 3-shingles (variant); CJK characters count as one
    token each, so mixed Chinese/English text is handled uniformly.
    Within-repo: EVERY pair of blocks from different episodes, via sparse matrix products.
    Cross-repo : every pair sharing >= 2 shingles of document frequency <= DF_CAP, plus
                 every identical-hash pair. (Exhaustive cross-repo is wasteful: Step 12
                 found ~2 qualifying cross-repo pairs.)
    Lexical bins (Step 12 rules, PROVISIONAL until human calibration):
        exact : identical normalized text
        near  : J5 >= 0.50, or C5 >= 0.80 with >= 10 shared shingles
        gray  : max(J5, C5) >= 0.30
        low   : the rest

Axis 2 - semantic:
    Sentence embeddings (default BAAI/bge-m3, multilingual), L2-normalized, cosine.
    Within-repo: every pair. Cross-repo: top-k neighbours per block in other repos.
    Semantic bins are relative to a NULL distribution: cosine of 300,000 random block
    pairs from different repository families (identical texts excluded).
        sem_high : cosine >= 99.9th percentile of the null
        sem_mid  : cosine >= 99th percentile
        sem_low  : the rest
    These are retrieval bins, NOT clone labels. Human labels decide what they mean.

Provenance (decides eligibility, never the similarity label):
    lineage           : same repo AND same capability / feature key / change id
    within_repo_reuse : same repo, otherwise
    shared_ancestry   : different repos, same Step 12 family
    same_owner        : different repos, same GitHub owner
    cross_repo        : different repos, otherwise

Episode-pair categories (from NON-lineage block pairs only):
    lineage_only       : every block pair is lineage (same capability/feature) - never a control
    copied             : >= 1 exact or near block pair
    gray               : else >= 1 gray block pair
    reworded_candidate : else >= 1 block pair that is lexically low AND sem_high
    related_candidate  : else >= 1 block pair that is lexically low AND sem_mid
    unrelated          : else
    has_lineage_match  : flag - some lineage block pair is exact/near (revision, not reuse)

Outputs (outputs/):
    12c_block_index.parquet          one row per block: row id, ids, sizes, truncation
    12c_within_block_pairs.parquet   within-repo block pairs with any signal, plus a
                                     random sample of the rest (column sample_weight)
    12c_within_episode_pairs.parquet every within-repo episode pair, aggregated metrics
    12c_cross_block_pairs.parquet    cross-repo candidate block pairs
    12c_cross_episode_pairs.parquet  cross-repo episode pairs, aggregated
    12c_summary.txt                  grids, Step 12 comparison, GATE 1 (RQ3 feasibility)
Cache:  cache/12c/emb/               embeddings in resumable chunks

Usage:
    python src\\12c_two_axis_similarity.py
    python src\\12c_two_axis_similarity.py --model intfloat/multilingual-e5-base   (faster)
    python src\\12c_two_axis_similarity.py --embedder hashing   (SMOKE TEST ONLY - not semantic)
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

SEED = 20261019
NEAR_J, NEAR_C, NEAR_C_MIN_SHARED, GRAY = 0.50, 0.80, 10, 0.30
DF_CAP = 500            # cross-repo candidate generation only
CROSS_MIN_SHARED = 2    # cross-repo candidate generation only
TOPK_CROSS = 10
NULL_PAIRS = 300_000
MAX_PAIRS_PER_CHUNK = 3_000_000   # bounds memory for very large repos
SAMPLE_RATE = 0.02      # random sample of no-signal within-repo pairs (weight 1/rate)
ROW_CHUNK = 1000
EMB_CHUNK = 2000

# signal floors for storing within-repo block pairs (storage only, not labels)
KEEP_J5, KEEP_C5, KEEP_J3 = 0.10, 0.30, 0.15

CJK_CLASS = r"぀-ヿ㐀-䶿一-鿿가-힯豈-﫿"
TOKEN_RE = re.compile(rf"[{CJK_CLASS}]|[^\s{CJK_CLASS}]+")

LEX_BINS = ["exact", "near", "gray", "low"]
SEM_BINS = ["sem_low", "sem_mid", "sem_high"]


def log(msg, t0):
    print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# lexical axis
# --------------------------------------------------------------------------- #
def shingle_hashes(text: str, n: int) -> set[int]:
    toks = TOKEN_RE.findall(text or "")
    if not toks:
        return set()
    if len(toks) < n:
        return {hash(" ".join(toks))}
    return {hash(" ".join(toks[k:k + n])) for k in range(len(toks) - n + 1)}


def build_matrix(texts: list[str], n: int) -> sparse.csr_matrix:
    vocab: dict[int, int] = {}
    indptr, indices = [0], []
    for t in texts:
        for h in shingle_hashes(t, n):
            indices.append(vocab.setdefault(h, len(vocab)))
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float32)
    X = sparse.csr_matrix((data, np.array(indices, dtype=np.int64), np.array(indptr, dtype=np.int64)),
                          shape=(len(texts), len(vocab)))
    X.sum_duplicates()
    return X


def jc(inter, si, sj):
    union = si + sj - inter
    jac = np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float64), where=union > 0)
    mn = np.minimum(si, sj)
    con = np.divide(inter, mn, out=np.zeros_like(inter, dtype=np.float64), where=mn > 0)
    return jac, con


def lex_code(exact, j5, c5, inter5):
    near = (j5 >= NEAR_J) | ((c5 >= NEAR_C) & (inter5 >= NEAR_C_MIN_SHARED))
    gray = np.maximum(j5, c5) >= GRAY
    return np.where(exact, 0, np.where(near, 1, np.where(gray, 2, 3))).astype(np.int8)


def rowwise_inter(X, ii, jj, batch=200_000):
    out = np.empty(len(ii), dtype=np.float64)
    for s in range(0, len(ii), batch):
        a, b = ii[s:s + batch], jj[s:s + batch]
        out[s:s + batch] = np.asarray(X[a].multiply(X[b]).sum(axis=1)).ravel()
    return out


# --------------------------------------------------------------------------- #
# semantic axis
# --------------------------------------------------------------------------- #
def embed_hashing(texts, dim=512):
    """SMOKE TEST ONLY: signed feature hashing of tokens. Not a semantic model."""
    E = np.zeros((len(texts), dim), dtype=np.float32)
    for r, t in enumerate(texts):
        for tok in TOKEN_RE.findall((t or "").lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
            E[r, h % dim] += 1.0 if (h >> 20) & 1 else -1.0
    nrm = np.linalg.norm(E, axis=1, keepdims=True)
    return E / np.maximum(nrm, 1e-9), np.zeros(len(texts), dtype=np.int32)


def embed_st(texts, model_name, max_len, batch, cache_dir: Path, t0):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"loading {model_name} on {device}", t0)
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_len
    prefix = "query: " if "e5" in model_name.lower() else ""
    texts = [prefix + (t or "") for t in texts]

    tok = model.tokenizer
    lens = np.zeros(len(texts), dtype=np.int32)
    for s in range(0, len(texts), 5000):
        enc = tok(texts[s:s + 5000], add_special_tokens=True, truncation=False)["input_ids"]
        lens[s:s + 5000] = [len(x) for x in enc]

    h = hashlib.sha1(f"{model_name}|{max_len}".encode())
    for t in texts:
        h.update(t.encode("utf-8", "ignore"))
        h.update(b"\x00")
    key = h.hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    n_chunks = (len(texts) + EMB_CHUNK - 1) // EMB_CHUNK
    for c, s in enumerate(range(0, len(texts), EMB_CHUNK)):
        f = cache_dir / f"{key}_{s:07d}.npy"
        if f.exists():
            parts.append(np.load(f))
            continue
        v = model.encode(texts[s:s + EMB_CHUNK], batch_size=batch, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        np.save(f, v)
        parts.append(v)
        log(f"embedded chunk {c + 1}/{n_chunks}", t0)
    E = np.vstack(parts)
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    return E, lens


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--cache", default="cache/12c")
    ap.add_argument("--variant", default="norm_text", help="text column for the lexical axis")
    ap.add_argument("--embed-col", default="norm_text", help="text column for the semantic axis")
    ap.add_argument("--embedder", default="st", choices=["st", "hashing"])
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out)
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # ---------------- load ---------------- #
    blocks = pd.read_parquet(out / "11_spec_blocks.parquet")
    for c in (args.variant, args.embed_col):
        if c not in blocks.columns:
            raise SystemExit(f"column '{c}' not in 11_spec_blocks.parquet; available: {list(blocks.columns)}")
    B = blocks[blocks["primary"] & blocks["substantive"] & ~blocks["dup_within_episode"]].reset_index(drop=True)
    man = pd.read_csv(out / "10b_episode_manifest.csv")
    man = man[man["status"] == "ok"]
    langs = {e: frozenset(x for x in str(l).split(";") if x and x != "nan")
             for e, l in zip(man["episode_id"], man["languages"])}
    fam_df = pd.read_csv(out / "12_repo_families.csv")
    fam = dict(zip(fam_df["repo"], fam_df["family"]))
    n = len(B)
    log(f"{n:,} primary blocks, {B['episode_id'].nunique():,} episodes, {B['repo'].nunique():,} repos", t0)

    B["uid"] = B["episode_id"].astype(str) + "|" + B["file_path"].astype(str) + "|" + B["block_idx"].astype(str)
    ep_codes, ep_names = pd.factorize(B["episode_id"])
    repo_codes, repo_names = pd.factorize(B["repo"])
    fam_codes = pd.factorize(B["repo"].map(lambda r: fam.get(r, r)))[0]
    owner_codes = pd.factorize(B["repo"].str.split("/").str[0])[0]
    hash_codes = pd.factorize(B["norm_hash"])[0]
    lineage_cols = [c for c in ("capability", "feature_key", "change_id") if c in B.columns]
    lin_codes = []
    for c in lineage_cols:
        v = B[c].replace("", np.nan)
        lin_codes.append(pd.factorize(v)[0])          # NaN -> -1
    log(f"lineage keys used: {lineage_cols}", t0)

    # ---------------- lexical matrices ---------------- #
    texts = B[args.variant].fillna("").tolist()
    X5 = build_matrix(texts, 5)
    X3 = build_matrix(texts, 3)
    s5 = np.asarray(X5.getnnz(axis=1), dtype=np.float64)
    s3 = np.asarray(X3.getnnz(axis=1), dtype=np.float64)
    log(f"shingle matrices: 5-gram vocab {X5.shape[1]:,}, 3-gram vocab {X3.shape[1]:,}", t0)

    # ---------------- semantic embeddings ---------------- #
    etexts = B[args.embed_col].fillna("").tolist()
    if args.embedder == "hashing":
        log("WARNING: hashing embedder - smoke test only, NOT semantic", t0)
        E, tok_lens = embed_hashing(etexts)
        model_label = "hashing (smoke test)"
    else:
        E, tok_lens = embed_st(etexts, args.model, args.max_len, args.batch, Path(args.cache) / "emb", t0)
        model_label = args.model
    truncated = tok_lens > args.max_len

    # null distribution: random pairs from different families
    ii = rng.integers(0, n, NULL_PAIRS * 2)
    jj = rng.integers(0, n, NULL_PAIRS * 2)
    ok = (fam_codes[ii] != fam_codes[jj]) & (hash_codes[ii] != hash_codes[jj])
    ii, jj = ii[ok][:NULL_PAIRS], jj[ok][:NULL_PAIRS]
    null = np.einsum("ij,ij->i", E[ii], E[jj])
    pct = {p: float(np.percentile(null, p)) for p in (50, 90, 99, 99.9)}
    T_MID, T_HIGH = pct[99], pct[99.9]
    log(f"null cosine: median {pct[50]:.3f}, p99 {T_MID:.3f}, p99.9 {T_HIGH:.3f} ({len(null):,} pairs)", t0)

    def sem_code(cos):
        return np.where(cos >= T_HIGH, 2, np.where(cos >= T_MID, 1, 0)).astype(np.int8)

    pd.DataFrame({
        "row": np.arange(n), "uid": B["uid"], "episode_id": B["episode_id"], "repo": B["repo"],
        "n_tokens": B["n_tokens"], "n_shingles5": s5.astype(int), "n_shingles3": s3.astype(int),
        "model_tokens": tok_lens, "truncated": truncated,
    }).to_parquet(out / "12c_block_index.parquet", index=False)

    # ---------------- within-repo: exhaustive ---------------- #
    grid_within = np.zeros((2, 4, 3), dtype=np.int64)        # [lineage?, lex, sem]
    kept_frames, agg_frames = [], []
    n_pairs_total = 0
    groups = pd.Series(np.arange(n)).groupby(repo_codes).apply(lambda s: s.to_numpy())
    order = sorted(groups.index, key=lambda g: -len(groups[g]))
    log(f"largest repos by blocks: " +
        ", ".join(f"{repo_names[g]}={len(groups[g])}" for g in order[:5]), t0)

    for gi, g in enumerate(order):
        idx = groups[g]
        if len(idx) < 2 or len(np.unique(ep_codes[idx])) < 2:
            continue
        Xr5, Xr3, Er = X5[idx], X3[idx], E[idx]
        epr = ep_codes[idx]
        linr = [lc[idx] for lc in lin_codes]
        chunk = int(max(50, min(ROW_CHUNK, MAX_PAIRS_PER_CHUNK // len(idx))))
        repo_frames = []
        for start in range(0, len(idx), chunk):
            a = np.arange(start, min(start + chunk, len(idx)))
            mask = (np.arange(len(idx))[None, :] > a[:, None]) & (epr[a][:, None] != epr[None, :])
            if not mask.any():
                continue
            li, lj = np.nonzero(mask)
            li = a[li]
            I5 = (Xr5[a] @ Xr5.T).toarray()
            I3 = (Xr3[a] @ Xr3.T).toarray()
            C = Er[a] @ Er.T
            r0 = li - start
            inter5, inter3, cos = I5[r0, lj].astype(np.float64), I3[r0, lj].astype(np.float64), C[r0, lj]
            del I5, I3, C
            gi_, gj_ = idx[li], idx[lj]
            j5, c5 = jc(inter5, s5[gi_], s5[gj_])
            j3, _ = jc(inter3, s3[gi_], s3[gj_])
            exact = hash_codes[gi_] == hash_codes[gj_]
            lx = lex_code(exact, j5, c5, inter5)
            sm = sem_code(cos)
            lin = np.zeros(len(li), dtype=bool)
            for lc in linr:
                lin |= (lc[li] == lc[lj]) & (lc[li] >= 0)
            np.add.at(grid_within, (lin.astype(int), lx, sm), 1)
            n_pairs_total += len(li)

            keep = exact | (j5 >= KEEP_J5) | (c5 >= KEEP_C5) | (j3 >= KEEP_J3) | (sm >= 1)
            samp = ~keep & (rng.random(len(li)) < SAMPLE_RATE)
            sel = keep | samp
            if sel.any():
                kept_frames.append(pd.DataFrame({
                    "i": gi_[sel], "j": gj_[sel], "repo": repo_names[g],
                    "inter5": inter5[sel].astype(np.int32), "j5": j5[sel].astype(np.float32),
                    "c5": c5[sel].astype(np.float32), "j3": j3[sel].astype(np.float32),
                    "cos": cos[sel].astype(np.float32), "lex_bin": lx[sel], "sem_bin": sm[sel],
                    "lineage": lin[sel], "sample_weight": np.where(keep[sel], 1.0, 1.0 / SAMPLE_RATE),
                }))

            e1, e2 = ep_codes[gi_], ep_codes[gj_]
            lo, hi = np.minimum(e1, e2), np.maximum(e1, e2)
            nl = ~lin
            repo_frames.append(pd.DataFrame({
                "e1": lo, "e2": hi,
                "max_j5": np.where(nl, j5, 0.0), "max_c5": np.where(nl, c5, 0.0),
                "max_j3": np.where(nl, j3, 0.0), "max_cos": np.where(nl, cos, -1.0),
                "max_cos_lexlow": np.where(nl & (lx == 3), cos, -1.0),
                "n_exact": (nl & (lx == 0)).astype(np.int32), "n_near": (nl & (lx == 1)).astype(np.int32),
                "n_gray": (nl & (lx == 2)).astype(np.int32),
                "n_semhigh_lexlow": (nl & (lx == 3) & (sm == 2)).astype(np.int32),
                "n_semmid_lexlow": (nl & (lx == 3) & (sm == 1)).astype(np.int32),
                "n_lineage_match": (lin & (lx <= 1)).astype(np.int32),
                "n_block_pairs": np.ones(len(li), dtype=np.int32),
                "n_nonlineage_pairs": nl.astype(np.int32),
            }).groupby(["e1", "e2"], sort=False).agg(AGG))
        if repo_frames:
            agg_frames.append(pd.concat(repo_frames).groupby(level=[0, 1]).agg(AGG))
        if gi % 25 == 0:
            log(f"within-repo: {gi + 1}/{len(order)} repos, {n_pairs_total:,} block pairs", t0)
    log(f"within-repo done: {n_pairs_total:,} block pairs compared exhaustively", t0)

    within_blocks = pd.concat(kept_frames, ignore_index=True) if kept_frames else pd.DataFrame()
    within_blocks.to_parquet(out / "12c_within_block_pairs.parquet", index=False)
    wep = finish_episode_pairs(agg_frames, ep_names, langs, B)
    wep.to_parquet(out / "12c_within_episode_pairs.parquet", index=False)

    # ---------------- cross-repo ---------------- #
    df5 = np.asarray(X5.getnnz(axis=0))
    Xcap = X5[:, np.nonzero(df5 <= DF_CAP)[0]].tocsr()
    cand = set()
    for start in range(0, n, 500):
        a = np.arange(start, min(start + 500, n))
        P = (Xcap[a] @ Xcap.T).tocoo()
        r = a[P.row]
        m = (P.col > r) & (P.data >= CROSS_MIN_SHARED) & (repo_codes[r] != repo_codes[P.col])
        cand.update(zip(r[m].tolist(), P.col[m].tolist()))
    for _, members in pd.Series(np.arange(n)).groupby(hash_codes):
        mem = members.to_numpy()
        if len(mem) > 1:
            for x in range(len(mem)):
                for y in range(x + 1, len(mem)):
                    if repo_codes[mem[x]] != repo_codes[mem[y]]:
                        cand.add((int(mem[x]), int(mem[y])))
    n_lex_cand = len(cand)
    for start in range(0, n, 1024):
        a = np.arange(start, min(start + 1024, n))
        S = E[a] @ E.T
        S[repo_codes[a][:, None] == repo_codes[None, :]] = -np.inf
        k = min(TOPK_CROSS, n - 1)
        top = np.argpartition(-S, k, axis=1)[:, :k]
        for r_local, cols in enumerate(top):
            i = int(a[r_local])
            for j in cols:
                if S[r_local, j] >= T_MID:
                    cand.add((min(i, int(j)), max(i, int(j))))
    log(f"cross-repo candidates: {n_lex_cand:,} lexical, {len(cand):,} with semantic top-{TOPK_CROSS}", t0)

    if cand:
        ci, cj = map(np.array, zip(*sorted(cand)))
        inter5 = rowwise_inter(X5, ci, cj)
        inter3 = rowwise_inter(X3, ci, cj)
        j5, c5 = jc(inter5, s5[ci], s5[cj])
        j3, _ = jc(inter3, s3[ci], s3[cj])
        cos = np.einsum("ij,ij->i", E[ci], E[cj])
        exact = hash_codes[ci] == hash_codes[cj]
        lx, sm = lex_code(exact, j5, c5, inter5), sem_code(cos)
        prov = np.where(fam_codes[ci] == fam_codes[cj], "shared_ancestry",
                        np.where(owner_codes[ci] == owner_codes[cj], "same_owner", "cross_repo"))
        cross = pd.DataFrame({
            "i": ci, "j": cj, "repo_i": repo_names[repo_codes[ci]], "repo_j": repo_names[repo_codes[cj]],
            "inter5": inter5.astype(np.int32), "j5": j5.astype(np.float32), "c5": c5.astype(np.float32),
            "j3": j3.astype(np.float32), "cos": cos.astype(np.float32), "lex_bin": lx, "sem_bin": sm,
            "provenance": prov,
        })
    else:
        cross = pd.DataFrame(columns=["i", "j", "repo_i", "repo_j", "inter5", "j5", "c5", "j3", "cos",
                                      "lex_bin", "sem_bin", "provenance"])
    cross.to_parquet(out / "12c_cross_block_pairs.parquet", index=False)

    elig = cross[cross["provenance"] != "shared_ancestry"] if len(cross) else cross
    cagg = []
    if len(elig):
        e1, e2 = ep_codes[elig["i"].to_numpy()], ep_codes[elig["j"].to_numpy()]
        lx, sm = elig["lex_bin"].to_numpy(), elig["sem_bin"].to_numpy()
        cagg.append(pd.DataFrame({
            "e1": np.minimum(e1, e2), "e2": np.maximum(e1, e2),
            "max_j5": elig["j5"].to_numpy(), "max_c5": elig["c5"].to_numpy(), "max_j3": elig["j3"].to_numpy(),
            "max_cos": elig["cos"].to_numpy(), "max_cos_lexlow": np.where(lx == 3, elig["cos"].to_numpy(), -1.0),
            "n_exact": (lx == 0).astype(np.int32), "n_near": (lx == 1).astype(np.int32),
            "n_gray": (lx == 2).astype(np.int32),
            "n_semhigh_lexlow": ((lx == 3) & (sm == 2)).astype(np.int32),
            "n_semmid_lexlow": ((lx == 3) & (sm == 1)).astype(np.int32),
            "n_lineage_match": np.zeros(len(elig), dtype=np.int32),
            "n_block_pairs": np.ones(len(elig), dtype=np.int32),
            "n_nonlineage_pairs": np.ones(len(elig), dtype=np.int32),
        }).groupby(["e1", "e2"]).agg(AGG))
    cep = finish_episode_pairs(cagg, ep_names, langs, B)
    cep.to_parquet(out / "12c_cross_episode_pairs.parquet", index=False)

    write_summary(out, args, model_label, n, B, truncated, pct, T_MID, T_HIGH, grid_within,
                  n_pairs_total, within_blocks, wep, cross, cep, t0)
    log("done", t0)


AGG = {"max_j5": "max", "max_c5": "max", "max_j3": "max", "max_cos": "max", "max_cos_lexlow": "max",
       "n_exact": "sum", "n_near": "sum", "n_gray": "sum", "n_semhigh_lexlow": "sum",
       "n_semmid_lexlow": "sum", "n_lineage_match": "sum", "n_block_pairs": "sum",
       "n_nonlineage_pairs": "sum"}


def finish_episode_pairs(frames, ep_names, langs, B):
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).groupby(level=[0, 1]).agg(AGG).reset_index()
    df["episode_i"] = ep_names[df["e1"].to_numpy()]
    df["episode_j"] = ep_names[df["e2"].to_numpy()]
    df["category"] = np.select(
        [df["n_nonlineage_pairs"] == 0, df["n_exact"] + df["n_near"] > 0, df["n_gray"] > 0, df["n_semhigh_lexlow"] > 0, df["n_semmid_lexlow"] > 0],
        ["lineage_only", "copied", "gray", "reworded_candidate", "related_candidate"], default="unrelated")
    df["has_lineage_match"] = df["n_lineage_match"] > 0
    shared = [sorted(langs.get(a, frozenset()) & langs.get(b, frozenset()))
              for a, b in zip(df["episode_i"], df["episode_j"])]
    df["shared_languages"] = [";".join(s) for s in shared]
    df["shared_language"] = [len(s) > 0 for s in shared]
    repo_of = dict(zip(B["episode_id"], B["repo"]))
    df["repo_i"] = df["episode_i"].map(repo_of)
    df["repo_j"] = df["episode_j"].map(repo_of)
    return df.drop(columns=["e1", "e2"])


def write_summary(out, args, model_label, n, B, truncated, pct, T_MID, T_HIGH, grid_within,
                  n_pairs_total, within_blocks, wep, cross, cep, t0):
    L = []
    w = L.append
    w("STEP 12c - TWO-AXIS SPEC SIMILARITY (exhaustive lexical x semantic)")
    w("=" * 72)
    w(f"Lexical text: {args.variant} | 5-shingles primary, 3-shingles variant")
    w(f"Semantic model: {model_label} | embed text: {args.embed_col} | max_len {args.max_len}")
    w(f"Blocks: {n:,} in {B['episode_id'].nunique():,} episodes, {B['repo'].nunique():,} repos")
    w(f"Blocks longer than max_len (tail truncated in embedding): {int(truncated.sum()):,} "
      f"({100 * truncated.mean():.1f}%)")
    w("")
    w("NULL COSINE (random pairs, different repo families)")
    w("  " + "  ".join(f"p{k}={v:.3f}" for k, v in pct.items()))
    w(f"  sem_mid >= {T_MID:.3f} (p99)   sem_high >= {T_HIGH:.3f} (p99.9)   [provisional]")
    w("")
    w(f"WITHIN-REPO BLOCK PAIRS (every pair, different episodes): {n_pairs_total:,}")
    for k, name in ((0, "NON-LINEAGE (within_repo_reuse)"), (1, "LINEAGE (revisions of one requirement)")):
        g = pd.DataFrame(grid_within[k], index=LEX_BINS, columns=SEM_BINS)
        g["All"] = g.sum(axis=1)
        w(f"\n  {name}")
        w("  " + g.to_string().replace("\n", "\n  "))
    w("\n  Compare with Step 12 (LSH): within_repo_reuse exact 18 / near 138 / gray 157;")
    w("  lineage exact 1405 / near 930. Exhaustive counts should be >= these;")
    w("  the excess is what LSH missed. (Shingling differs slightly from Step 12.)")

    w("\nWITHIN-REPO EPISODE PAIRS")
    if len(wep):
        prim = wep[~wep["has_lineage_match"]]
        w(f"  all episode pairs: {len(wep):,}; with a lineage match (excluded below): "
          f"{int(wep['has_lineage_match'].sum()):,}")
        ct = pd.crosstab(prim["category"], prim["shared_language"], margins=True)
        w("  category x shared_language")
        w("  " + ct.to_string().replace("\n", "\n  "))
        w("\n  repos contributing (shared language only)")
        sl = prim[prim["shared_language"]]
        w("  " + sl.groupby("category")["repo_i"].nunique().to_string().replace("\n", "\n  "))

        w("\nGATE 1 - RQ3 FEASIBILITY (shared language, no lineage match)")
        for cat in ("copied", "gray", "reworded_candidate"):
            s = sl[sl["category"] == cat]
            top = s["repo_i"].value_counts().head(3)
            share = 100 * top.iloc[0] / len(s) if len(s) else 0
            w(f"  {cat:20s}: {len(s):5d} episode pairs from {s['repo_i'].nunique():3d} repos"
              f"   largest repo share {share:.0f}%")
        w("  Rule of thumb: if reworded_candidate < ~20 pairs or ~10 repos even BEFORE human")
        w("  labelling removes false positives, RQ3 becomes a descriptive case analysis.")

    w("\nCROSS-REPO BLOCK PAIRS (candidates: shared rare shingles, identical text, semantic top-k)")
    if len(cross):
        c = cross.copy()
        c["lex"] = c["lex_bin"].map(dict(enumerate(LEX_BINS)))
        c["sem"] = c["sem_bin"].map(dict(enumerate(SEM_BINS)))
        for p in ("cross_repo", "same_owner", "shared_ancestry"):
            s = c[c["provenance"] == p]
            if len(s):
                w(f"\n  {p}")
                w("  " + pd.crosstab(s["lex"], s["sem"], margins=True).to_string().replace("\n", "\n  "))
    if len(cep):
        w("\n  cross-repo episode pairs (shared ancestry excluded): " +
          ", ".join(f"{k}={v}" for k, v in cep["category"].value_counts().items()))

    w("\nAll thresholds are PROVISIONAL. Do not look at code-clone outcomes before they are frozen.")
    w(f"Runtime {time.time() - t0:.0f}s")
    text = "\n".join(L)
    (out / "12c_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
    