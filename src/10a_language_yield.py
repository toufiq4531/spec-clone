"""
Step 10a - Language yield for the RQ2 episode set
=================================================

Question this answers:
    Among merged OpenSpec / SpecKit PRs that change BOTH specification files and
    production code, which programming languages have enough episodes and
    repositories to support same-language spec-pair -> code-clone comparisons?

Why: RQ2 compares code between two episodes, so every pair must share a language
and the clone detector must parse that language. This output decides which
languages enter the primary analysis (and therefore which detector setup is needed).

Usage (PowerShell, from the project root D:\\USask\\specmine\\clones-before-code):
    python src\\10a_language_yield.py --data D:\\USask\\specmine\\datasets\\specmine-full

Outputs:
    outputs\\10a_language_yield.txt   human-readable summary
    outputs\\10a_language_yield.csv   per-language table
    outputs\\10a_episode_languages.csv per-(PR, language) rows for later steps

Notes:
    * Column names are auto-detected; if one cannot be found the script stops and
      prints the available columns so you can add the right name to CANDIDATES.
    * Language is taken from changed code-file extensions, NOT repository language.
    * Tests, vendored, generated, minified and type-declaration files are excluded
      from the primary count (tests are reported separately).
"""

from __future__ import annotations

import argparse
import re
import sys
from math import comb
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq  # no DuckDB: its DLL is blocked by Windows App Control on this machine

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

EXT_LANG = {
    ".py": "Python",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".mts": "TypeScript", ".cts": "TypeScript",
    ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".kt": "Kotlin", ".kts": "Kotlin",
    ".cs": "C#",
    ".c": "C",
    ".cpp": "C++", ".cc": "C++", ".cxx": "C++", ".hpp": "C++", ".hh": "C++",
    ".h": "C/C++ header",
    ".rb": "Ruby",
    ".php": "PHP",
    ".swift": "Swift",
    ".ex": "Elixir", ".exs": "Elixir",
    ".dart": "Dart",
    ".scala": "Scala",
    ".vue": "Vue",
    ".svelte": "Svelte",
}

# Path patterns (matched on lower-cased, forward-slash paths)
TEST_RE = re.compile(
    r"(^|/)(tests?|__tests__|testdata|fixtures?|e2e)/"
    r"|_test\.(go|py|rs)$"
    r"|(^|/)test_[^/]*\.py$"
    r"|\.(test|spec)\.[a-z]+$"
)
EXCLUDE_RE = re.compile(
    r"(^|/)(vendor|node_modules|third_party|dist|build|out|target|\.next|coverage"
    r"|generated|__generated__|gen)/"
    r"|\.min\.js$|\.d\.ts$|\.pb\.go$|_pb2(_grpc)?\.py$|\.generated\.[a-z]+$"
)

TOOLS_KEPT = {"openspec": "OpenSpec", "speckit": "SpecKit"}

CANDIDATES = {
    # pull_requests table
    "pr_repo":   ["repo_full_name", "full_name", "repository", "repo", "repo_name", "nameWithOwner"],
    "pr_number": ["pr_number", "number", "pull_number", "pr_num"],
    "pr_id":     ["pr_id", "pull_request_id", "pr_uid"],
    "pr_tool":   ["tool", "sdd_tool", "tool_name", "tool_family"],
    "pr_merged": ["merged", "is_merged", "merged_at", "merge_status", "state"],
    # pr_files table
    "f_repo":    ["repo_full_name", "full_name", "repository", "repo", "repo_name", "nameWithOwner"],
    "f_number":  ["pr_number", "number", "pull_number", "pr_num"],
    "f_id":      ["pr_id", "pull_request_id", "pr_uid"],
    "f_path":    ["path", "filename", "file_path", "file", "new_path"],
    "f_is_spec": ["is_spec", "spec", "is_spec_file"],
    "f_is_code": ["is_code", "code", "is_code_file"],
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def find_parquet(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if not hits:
        sys.exit(f"[error] {name} not found under {root}")
    return hits[0]


def columns(path: Path) -> list[str]:
    return list(pq.read_schema(path).names)


def load(path: Path, cols: list[str]) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(dict.fromkeys(cols)), engine="pyarrow")


def pick(cols: list[str], key: str, required: bool = True) -> str | None:
    lower = {c.lower(): c for c in cols}
    for cand in CANDIDATES[key]:
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        sys.exit(f"[error] could not find a column for '{key}'.\n"
                 f"        Tried: {CANDIDATES[key]}\n"
                 f"        Available: {cols}\n"
                 f"        Add the right name to CANDIDATES['{key}'] and rerun.")
    return None


def as_flag(s: pd.Series) -> pd.Series:
    if s.dtype == object or str(s.dtype).startswith(("str", "string")):
        return s.astype(str).str.lower().isin({"1", "true", "t", "yes", "y"})
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int) > 0


def norm_tool(x: object) -> str | None:
    key = re.sub(r"[^a-z]", "", str(x).lower())
    for k, v in TOOLS_KEPT.items():
        if k in key:
            return v
    return None


def merged_mask(s: pd.Series, colname: str) -> pd.Series:
    name = colname.lower()
    if name == "merged_at":
        return s.notna() & (s.astype(str).str.strip() != "")
    if name in {"merge_status", "state"}:
        return s.astype(str).str.lower().str.contains("merged")
    return as_flag(s)


def language_of(path: str) -> str | None:
    p = path.lower()
    dot = p.rfind(".")
    if dot == -1:
        return None
    return EXT_LANG.get(p[dot:])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="SpecMine release root")
    ap.add_argument("--out", default="outputs", help="output folder")
    args = ap.parse_args()

    root = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pr_path = find_parquet(root, "pull_requests.parquet")
    f_path = find_parquet(root, "pr_files.parquet")
    pr_cols, f_cols = columns(pr_path), columns(f_path)

    # ---- choose join key -------------------------------------------------- #
    pr_id, f_id = pick(pr_cols, "pr_id", False), pick(f_cols, "f_id", False)
    pr_repo, pr_num = pick(pr_cols, "pr_repo"), pick(pr_cols, "pr_number")
    if pr_id and f_id:
        join = "id"
    else:
        f_repo, f_num = pick(f_cols, "f_repo"), pick(f_cols, "f_number")
        join = "repo_number"

    pr_tool = pick(pr_cols, "pr_tool")
    pr_merged = pick(pr_cols, "pr_merged", False)
    f_pathc = pick(f_cols, "f_path")
    f_spec = pick(f_cols, "f_is_spec")
    f_code = pick(f_cols, "f_is_code", False)

    # ---- load PRs --------------------------------------------------------- #
    sel = [pr_repo, pr_num, pr_tool] + ([pr_merged] if pr_merged else []) + ([pr_id] if join == "id" else [])
    prs = load(pr_path, sel)
    n_prs_total = len(prs)

    prs["tool_norm"] = prs[pr_tool].map(norm_tool)
    if pr_merged:
        prs = prs[merged_mask(prs[pr_merged], pr_merged)]
    n_merged = len(prs)
    prs = prs[prs["tool_norm"].notna()].copy()
    prs["repo"] = prs[pr_repo].astype(str)
    prs["pr"] = prs["repo"] + "#" + prs[pr_num].astype(str)
    n_tool = len(prs)

    # ---- load PR files ---------------------------------------------------- #
    if join == "id":
        fsel = [f_id, f_pathc, f_spec] + ([f_code] if f_code else [])
    else:
        fsel = [f_repo, f_num, f_pathc, f_spec] + ([f_code] if f_code else [])
    files = load(f_path, fsel)

    if join == "id":
        files = files.merge(prs[[pr_id, "pr", "repo", "tool_norm"]], left_on=f_id, right_on=pr_id, how="inner")
    else:
        files["pr"] = files[f_repo].astype(str) + "#" + files[f_num].astype(str)
        files = files.merge(prs[["pr", "repo", "tool_norm"]], on="pr", how="inner")

    files["path_l"] = files[f_pathc].astype(str).str.replace("\\", "/", regex=False).str.lower()
    files["is_spec_f"] = as_flag(files[f_spec])
    files["lang"] = files["path_l"].map(language_of)
    if f_code:
        files["is_code_f"] = as_flag(files[f_code]) | files["lang"].notna()
    else:
        files["is_code_f"] = files["lang"].notna()
    files["is_test"] = files["path_l"].str.contains(TEST_RE)
    files["is_excl"] = files["path_l"].str.contains(EXCLUDE_RE)

    spec_prs = set(files.loc[files["is_spec_f"], "pr"])
    prod = files[files["is_code_f"] & ~files["is_spec_f"] & files["lang"].notna()
                 & ~files["is_test"] & ~files["is_excl"]]
    tests = files[files["is_code_f"] & files["lang"].notna() & files["is_test"] & ~files["is_excl"]]

    prod = prod[prod["pr"].isin(spec_prs)]
    tests = tests[tests["pr"].isin(spec_prs)]

    ep_lang = (prod.groupby(["pr", "repo", "tool_norm", "lang"], as_index=False)
                   .agg(n_code_files=("path_l", "nunique")))
    ep_lang.to_csv(out / "10a_episode_languages.csv", index=False)

    # dominant language per PR
    dom = (ep_lang.sort_values(["pr", "n_code_files"], ascending=[True, False])
                  .drop_duplicates("pr"))
    n_episodes = ep_lang["pr"].nunique()
    n_multi = (ep_lang.groupby("pr")["lang"].nunique() > 1).sum()

    # ---- per-language table ---------------------------------------------- #
    rows = []
    for lang, g in ep_lang.groupby("lang"):
        per_repo = g.groupby("repo")["pr"].nunique()
        n = int(per_repo.sum())
        within = int(sum(comb(int(k), 2) for k in per_repo))
        total_pairs = comb(n, 2)
        rows.append({
            "language": lang,
            "episodes": n,
            "dominant_lang_episodes": int((dom["lang"] == lang).sum()),
            "repositories": int(per_repo.size),
            "repos_with_2plus_episodes": int((per_repo >= 2).sum()),
            "openspec_episodes": int(g.loc[g["tool_norm"] == "OpenSpec", "pr"].nunique()),
            "speckit_episodes": int(g.loc[g["tool_norm"] == "SpecKit", "pr"].nunique()),
            "median_code_files": float(g["n_code_files"].median()),
            "max_share_one_repo": round(float(per_repo.max() / n), 3) if n else 0.0,
            "within_repo_pairs": within,
            "cross_repo_pairs": total_pairs - within,
            "test_only_files_episodes": int(tests.loc[tests["lang"] == lang, "pr"].nunique()),
        })
    table = pd.DataFrame(rows).sort_values("episodes", ascending=False)
    table.to_csv(out / "10a_language_yield.csv", index=False)

    # ---- text summary ----------------------------------------------------- #
    lines = [
        "STEP 10a - LANGUAGE YIELD FOR RQ2 EPISODES",
        "=" * 60,
        f"pull_requests.parquet : {pr_path}",
        f"pr_files.parquet      : {f_path}",
        f"Join key used         : {join}",
        "",
        "FUNNEL",
        f"  PRs in table                                   : {n_prs_total:,}",
        f"  Merged PRs{' (no merge column found)' if not pr_merged else ''}"
        f"{'':<37}: {n_merged:,}",
        f"  ...of which OpenSpec or SpecKit                : {n_tool:,}",
        f"  ...with spec file AND production code file     : {n_episodes:,}",
        f"  ...touching more than one production language : {n_multi:,}",
        "",
        "Excluded from production code: tests (reported separately), vendor/node_modules,",
        "dist/build/target, generated dirs, *.min.js, *.d.ts, *.pb.go, *_pb2.py.",
        "",
        "PER-LANGUAGE YIELD (an episode can count under several languages)",
        table.to_string(index=False),
        "",
        "HOW TO READ THIS",
        "  episodes                 : PRs that change production code in this language",
        "  repos_with_2plus_episodes: repos that can contribute within-repo pairs",
        "  max_share_one_repo       : >0.3 means one repo dominates -> weak independence",
        "  within/cross_repo_pairs  : candidate pairs BEFORE spec-similarity filtering;",
        "                             spec-clone pairs will be a small fraction of these",
        "",
        "Decision input: choose 1-3 languages with the most episodes AND repositories,",
        "then confirm the clone detector has a parser for each.",
    ]
    text = "\n".join(lines)
    (out / "10a_language_yield.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()