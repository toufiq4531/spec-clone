"""
Step 12 - Specification clone pairs, provenance, and the yield gate
===================================================================

Input : outputs/11_spec_blocks.parquet  (primary, substantive, de-duplicated blocks)
        outputs/10b_episode_manifest.csv (episode languages)
        cache/10b/repos/*.git            (root commits for ancestry)

1. Block-level candidates. Word 5-shingles (CJK: character 5-shingles) per block;
   numpy MinHash (126 permutations, 42 bands x 3 rows, ~0.29 Jaccard LSH threshold).
   Exact Jaccard and containment (|A n B| / min(|A|,|B|)) are then computed for every
   candidate pair. Identical normalized blocks are always candidates.
2. Block-pair labels (PROVISIONAL - calibrated on the Step 14 development set, then frozen):
        exact : identical normalized text
        near  : Jaccard >= 0.50, or containment >= 0.80 with >= 10 shared shingles
        gray  : 0.30 <= max(Jaccard, containment) below the near rule
        low   : retrieved but below 0.30
3. Provenance of each pair (decides eligibility, never the similarity label):
        same_requirement_lineage : same repo AND same OpenSpec capability / change id or
                                   same SpecKit feature directory (revisions, archive/sync
                                   copies of ONE requirement - not reuse)
        shared_ancestry          : different repos sharing a root commit (fork / template /
                                   copied history)
        same_owner               : different repos, same owner (kept; sensitivity flag)
        within_repo_reuse        : same repo, different feature/capability
        cross_repo               : different repos, no shared root commit
4. Episode pairs: max block Jaccard / containment, number of near/exact block matches,
   symmetric block coverage (share of both episodes' substantive tokens that sit in
   blocks with a near/exact match in the other episode), shared languages.
   Exposure (provisional): A = >=1 exact/near block pair with eligible provenance.
5. Yield gate and an RQ1 preview.

Usage:   python src\\12_spec_pairs.py
Outputs: outputs\\12_block_pairs.parquet, outputs\\12_episode_pairs.parquet,
         outputs\\12_repo_families.csv, outputs\\12_yield_summary.txt
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

SHINGLE = 5
N_PERM, BANDS, ROWS = 126, 42, 3
MAX_BUCKET = 400                      # buckets larger than this are logged and skipped
NEAR_J, NEAR_C, NEAR_C_MIN_SHARED, GRAY = 0.50, 0.80, 10, 0.30
PRIME = np.uint64((1 << 61) - 1)
SEED = 20261019


def safe(repo: str) -> str:
    return repo.replace("/", "__")


def shingles(text: str) -> np.ndarray:
    toks = text.split()
    if len(toks) < SHINGLE:
        grams = [" ".join(toks)]
    else:
        grams = [" ".join(toks[i:i + SHINGLE]) for i in range(len(toks) - SHINGLE + 1)]
    return np.unique(np.fromiter(
        (int.from_bytes(hashlib.blake2b(g.encode(), digest_size=4).digest(), "little") for g in grams),
        dtype=np.uint64))


def minhash(sh: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a[:, None] * sh[None, :] + b[:, None]) % PRIME).min(axis=1)


def root_commits(repo_dir: Path) -> set[str]:
    if not repo_dir.exists():
        return set()                    # no local clone: ancestry unknown, repo is its own family
    p = subprocess.run(["git", "rev-list", "--max-parents=0", "HEAD"], cwd=repo_dir,
                       capture_output=True, timeout=600)
    return set(p.stdout.decode().split()) if p.returncode == 0 else set()


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--cache", default="cache/10b")
    ap.add_argument("--variant", default="norm_text", choices=["norm_text", "norm_text_with_code"])
    args = ap.parse_args()
    out = Path(args.out)
    t0 = time.time()

    blocks = pd.read_parquet(out / "11_spec_blocks.parquet")
    B = blocks[blocks["primary"] & blocks["substantive"] & ~blocks["dup_within_episode"]].reset_index(drop=True)
    man = pd.read_csv(out / "10b_episode_manifest.csv")
    man = man[man["status"] == "ok"]
    langs = {e: set(str(l).split(";")) for e, l in zip(man["episode_id"], man["languages"])}
    tool = dict(zip(man["episode_id"], man["tool"]))
    print(f"[load] {len(B):,} primary blocks in {B['episode_id'].nunique():,} episodes", flush=True)

    # ---------------- ancestry families ---------------- #
    repos = sorted(B["repo"].unique())
    uf, roots = UF(), {}
    root_owner = defaultdict(set)
    for r in repos:
        rc = root_commits(Path(args.cache) / "repos" / f"{safe(r)}.git")
        roots[r] = rc
        uf.find(r)
        for c in rc:
            root_owner[c].add(r)
    for c, rs in root_owner.items():
        rs = sorted(rs)
        for x in rs[1:]:
            uf.union(rs[0], x)
    fam = {r: uf.find(r) for r in repos}
    pd.DataFrame({"repo": repos, "family": [fam[r] for r in repos],
                  "n_root_commits": [len(roots[r]) for r in repos]}).to_csv(out / "12_repo_families.csv", index=False)
    n_shared = sum(1 for r in repos if fam[r] != r or sum(fam[x] == fam[r] for x in repos) > 1)
    print(f"[ancestry] {len(repos)} repos, {len(set(fam.values()))} families, "
          f"{n_shared} repos share history with another", flush=True)

    # ---------------- MinHash + LSH ---------------- #
    rng = np.random.default_rng(SEED)
    a = rng.integers(1, 1 << 31, N_PERM, dtype=np.uint64)
    b = rng.integers(0, 1 << 31, N_PERM, dtype=np.uint64)
    texts = B[args.variant].tolist()
    SH = [shingles(t) for t in texts]
    SIG = np.vstack([minhash(s, a, b) for s in SH])
    print(f"[minhash] signatures done ({time.time() - t0:.0f}s)", flush=True)

    ep = B["episode_id"].to_numpy()
    cand: set[tuple[int, int]] = set()
    skipped_buckets = 0
    for band in range(BANDS):
        keys = SIG[:, band * ROWS:(band + 1) * ROWS]
        buckets = defaultdict(list)
        for i, k in enumerate(map(bytes, keys)):
            buckets[k].append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            if len(members) > MAX_BUCKET:
                skipped_buckets += 1
                continue
            for i, j in combinations(members, 2):
                if ep[i] != ep[j]:
                    cand.add((i, j) if i < j else (j, i))
    # identical texts are always candidates (also inside oversized buckets)
    for _, idx in B.groupby("norm_hash").groups.items():
        idx = list(idx)
        for i, j in combinations(idx, 2):
            if ep[i] != ep[j]:
                cand.add((min(i, j), max(i, j)))
    print(f"[lsh] {len(cand):,} candidate block pairs, {skipped_buckets} oversized buckets skipped "
          f"({time.time() - t0:.0f}s)", flush=True)

    # ---------------- exact scoring ---------------- #
    sets = [set(s.tolist()) for s in SH]
    rows = []
    hashes = B["norm_hash"].to_numpy()
    for i, j in cand:
        si, sj = sets[i], sets[j]
        inter = len(si & sj)
        jac = inter / len(si | sj)
        con = inter / min(len(si), len(sj))
        exact = hashes[i] == hashes[j]
        if exact:
            label = "exact"
        elif jac >= NEAR_J or (con >= NEAR_C and inter >= NEAR_C_MIN_SHARED):
            label = "near"
        elif max(jac, con) >= GRAY:
            label = "gray"
        else:
            label = "low"
        rows.append((i, j, inter, round(jac, 4), round(con, 4), label))
    P = pd.DataFrame(rows, columns=["i", "j", "shared_shingles", "jaccard", "containment", "label"])

    def col(name):
        return B[name].to_numpy()
    for side in ("i", "j"):
        idx = P[side].to_numpy()
        for c in ("episode_id", "repo", "tool", "file_path", "doc_type", "feature_key", "capability",
                  "block_kind", "title", "n_tokens", "norm_hash"):
            P[f"{c}_{side}"] = col(c)[idx]

    def provenance(r) -> str:
        if r.repo_i == r.repo_j:
            same = ((isinstance(r.capability_i, str) and r.capability_i == r.capability_j) or
                    (isinstance(r.feature_key_i, str) and r.feature_key_i == r.feature_key_j))
            return "same_requirement_lineage" if same else "within_repo_reuse"
        if fam[r.repo_i] == fam[r.repo_j]:
            return "shared_ancestry"
        if r.repo_i.split("/")[0].lower() == r.repo_j.split("/")[0].lower():
            return "same_owner"
        return "cross_repo"
    P["provenance"] = [provenance(r) for r in P.itertuples(index=False)] if len(P) else []
    P.to_parquet(out / "12_block_pairs.parquet", index=False)
    print(f"[score] block pairs labelled ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- episode pairs ---------------- #
    tok = B.groupby("episode_id")["n_tokens"].sum().to_dict()
    clone = P[P["label"].isin(["exact", "near"])]
    ep_rows = []
    for (ei, ej), g in P.groupby(["episode_id_i", "episode_id_j"]):
        c = g[g["label"].isin(["exact", "near"])]
        cov_tokens = (c.drop_duplicates("i")["n_tokens_i"].sum() + c.drop_duplicates("j")["n_tokens_j"].sum())
        prov = g["provenance"]
        # an episode pair is lineage-only if every clone block pair is lineage
        if len(c):
            elig = c[~c["provenance"].isin(["same_requirement_lineage", "shared_ancestry"])]
        else:
            elig = c
        shared = sorted(langs.get(ei, set()) & langs.get(ej, set()))
        ep_rows.append({
            "episode_i": ei, "episode_j": ej, "repo_i": g["repo_i"].iloc[0], "repo_j": g["repo_j"].iloc[0],
            "tool_pair": "+".join(sorted([tool.get(ei, "?"), tool.get(ej, "?")])),
            "stratum": "within_repo" if g["repo_i"].iloc[0] == g["repo_j"].iloc[0] else "cross_repo",
            "max_jaccard": g["jaccard"].max(), "max_containment": g["containment"].max(),
            "n_exact": int((g["label"] == "exact").sum()), "n_near": int((g["label"] == "near").sum()),
            "n_clone_eligible": len(elig),
            "spec_coverage": round(cov_tokens / (tok[ei] + tok[ej]), 4),
            "provenance": ("lineage_only" if len(c) and elig.empty else
                           prov[~prov.isin(["same_requirement_lineage"])].mode().iloc[0]
                           if (~prov.isin(["same_requirement_lineage"])).any() else "same_requirement_lineage"),
            "shared_langs": ";".join(shared),
            "exposure": ("A" if len(elig) else
                         "lineage_only" if len(c) else
                         "gray" if (g["label"] == "gray").any() else "low"),
        })
    E = pd.DataFrame(ep_rows)
    E.to_parquet(out / "12_episode_pairs.parquet", index=False)

    # ---------------- RQ1 preview + yield gate ---------------- #
    eligible_eps = set(B["episode_id"])
    elig_clone = clone[~clone["provenance"].isin(["same_requirement_lineage", "shared_ancestry"])]
    eps_with = set(elig_clone["episode_id_i"]) | set(elig_clone["episode_id_j"])
    eps_cross = set(elig_clone.loc[elig_clone["repo_i"] != elig_clone["repo_j"], "episode_id_i"]) | \
        set(elig_clone.loc[elig_clone["repo_i"] != elig_clone["repo_j"], "episode_id_j"])
    A = E[(E["exposure"] == "A") & (E["shared_langs"] != "")] if len(E) else E

    def gate(sub: pd.DataFrame, stratum: str) -> list[str]:
        if sub.empty:
            return [f"  {stratum}: 0 A pairs"]
        if stratum == "within_repo":
            units = sub["repo_i"].nunique()
            unit_name = "repositories"
        else:
            g = UF()
            for r in sub.itertuples(index=False):
                g.union(fam[r.repo_i], fam[r.repo_j])
            units = len({g.find(fam[x]) for x in pd.concat([sub["repo_i"], sub["repo_j"]])})
            unit_name = "independent components"
        out_l = [f"  {stratum}: {len(sub):,} A pairs from {units} {unit_name} "
                 f"[gate {'PASS' if len(sub) >= 50 and units >= 15 else 'NOT MET'}: >=50 pairs, >=15 units]"]
        for lang in ("TypeScript", "Python", "Go"):
            s = sub[sub["shared_langs"].str.contains(lang)]
            out_l.append(f"      {lang:<11} {len(s):>6,} pairs, {s['repo_i'].nunique():>4} repos")
        return out_l

    lines = [
        "STEP 12 - SPEC CLONE PAIRS AND YIELD GATE",
        "=" * 60,
        f"Variant: {args.variant} | shingle={SHINGLE} | MinHash {N_PERM} perms, {BANDS}x{ROWS} bands",
        f"Provisional labels: near = J>={NEAR_J} or C>={NEAR_C} (>= {NEAR_C_MIN_SHARED} shared); gray >= {GRAY}",
        f"Oversized LSH buckets skipped: {skipped_buckets}",
        "",
        f"Primary blocks: {len(B):,} in {len(eligible_eps):,} episodes, {B['repo'].nunique()} repos",
        f"Repository families (shared root commits): {len(set(fam.values()))}",
        f"Candidate block pairs scored: {len(P):,}",
        "",
        "BLOCK PAIRS: LABEL x PROVENANCE",
        pd.crosstab(P["label"], P["provenance"], margins=True).to_string() if len(P) else "  none",
        "",
        "RQ1 PREVIEW (provisional thresholds, lineage and shared-ancestry pairs excluded)",
        f"  episodes with >=1 duplicated requirement block      : {len(eps_with):,} / {len(eligible_eps):,} "
        f"({len(eps_with) / max(len(eligible_eps), 1):.1%})",
        f"  ...duplicated in another repository                 : {len(eps_cross):,} "
        f"({len(eps_cross) / max(len(eligible_eps), 1):.1%})",
        "",
        "EPISODE PAIRS BY EXPOSURE x STRATUM",
        pd.crosstab(E["exposure"], E["stratum"], margins=True).to_string() if len(E) else "  none",
        "",
        "YIELD GATE (A pairs that share a primary language)",
        *gate(A[A["stratum"] == "within_repo"] if len(A) else A, "within_repo"),
        *gate(A[A["stratum"] == "cross_repo"] if len(A) else A, "cross_repo"),
        "",
        "PROVENANCE OF CROSS-REPO A PAIRS",
        A[A["stratum"] == "cross_repo"]["provenance"].value_counts().to_string() if len(A) else "  none",
        "",
        "LARGEST CONTRIBUTORS TO WITHIN-REPO A PAIRS (dominance check)",
        A[A["stratum"] == "within_repo"]["repo_i"].value_counts().head(10).to_string() if len(A) else "  none",
        "",
        f"Runtime: {time.time() - t0:.0f}s",
    ]
    text = "\n".join(lines)
    (out / "12_yield_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()