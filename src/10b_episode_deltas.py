"""
Step 10b - Episode manifest and spec/code deltas from git
=========================================================

For every RQ2 episode (merged OpenSpec/SpecKit PR that changes spec files AND
production code in a primary language), recover the exact revisions and the
changed files, and cache the before/after content of every relevant file.

What it does, per repository:
  1. Bare partial clone (commits + trees, no file contents):
        git clone --bare --filter=blob:none https://github.com/<repo>.git
  2. One GitHub API call per PR (cached) for head.sha, base.sha, merge_commit_sha.
  3. Fetch refs/pull/<N>/head (and base.sha if not already present).
  4. Diff base = git merge-base(base.sha, head.sha)  -> same base GitHub uses
     for the PR "Files changed" view. Squash/rebase merges are handled because
     we never rely on the merge commit or on SHA joins with SpecMine tables.
  5. git diff --raw -M  merge_base..head  -> changed paths, statuses, blob ids.
  6. Keep: spec files (is_spec in pr_files), production code and tests in the
     selected languages. Fetch ONLY those blobs (old and new versions).
  7. Save the patch for the selected paths (used later for annotation packets).

Resumable: each PR writes cache/10b/records/<owner>__<repo>__<N>.json; rerunning
skips finished PRs. Raw API responses are kept under cache/10b/api/.

Usage (PowerShell, from D:\\USask\\specmine\\clones-before-code):
    $env:GITHUB_TOKEN = "<your token>"          # needed: ~3,000 API calls
    # dry run on 3 repositories first:
    python src\\10b_episode_deltas.py --data D:\\USask\\specmine\\datasets\\specmine-full --max-repos 3
    # full run:
    python src\\10b_episode_deltas.py --data D:\\USask\\specmine\\datasets\\specmine-full --workers 6
    # rebuild CSVs/summary from cached records without network:
    python src\\10b_episode_deltas.py --data D:\\USask\\specmine\\datasets\\specmine-full --summarize-only

Inputs:
    outputs\\10a_episode_languages.csv   (from Step 10a)
    <data>\\data\\pull_requests.parquet, <data>\\data\\pr_files.parquet

Outputs:
    outputs\\10b_episode_manifest.csv    one row per episode (revisions, status, QA flags)
    outputs\\10b_episode_files.csv       one row per selected changed file
    outputs\\10b_summary.txt             acquisition funnel
    cache\\10b\\repos\\<owner>__<repo>.git  bare partial clones
    cache\\10b\\blobs\\<aa>\\<oid>         file contents, deduplicated by git blob id
    cache\\10b\\patches\\<owner>__<repo>__<N>.diff
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration (kept identical to Step 10a so the populations match)
# --------------------------------------------------------------------------- #

LANGS_DEFAULT = ["TypeScript", "Python", "Go"]

EXT_LANG = {
    ".py": "Python",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".mts": "TypeScript", ".cts": "TypeScript",
    ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".go": "Go", ".rs": "Rust", ".java": "Java", ".kt": "Kotlin", ".kts": "Kotlin",
    ".cs": "C#", ".c": "C", ".cpp": "C++", ".cc": "C++", ".cxx": "C++", ".hpp": "C++",
    ".hh": "C++", ".h": "C/C++ header", ".rb": "Ruby", ".php": "PHP", ".swift": "Swift",
    ".ex": "Elixir", ".exs": "Elixir", ".dart": "Dart", ".scala": "Scala",
    ".vue": "Vue", ".svelte": "Svelte",
}
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

ZERO = "0" * 40
MAX_BLOB_BYTES = 2_000_000          # larger files are almost always generated/data
PATHSPEC_CHARS = 20_000             # keep command lines under the Windows limit
GIT_ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def as_flag(s: pd.Series) -> pd.Series:
    if s.dtype == object or str(s.dtype).startswith(("str", "string")):
        return s.astype(str).str.lower().isin({"1", "true", "t", "yes", "y"})
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int) > 0


def language_of(path: str) -> str | None:
    p = path.lower()
    dot = p.rfind(".")
    return EXT_LANG.get(p[dot:]) if dot != -1 else None


def safe(repo: str) -> str:
    return repo.replace("/", "__")


class GitError(RuntimeError):
    pass


def git(args: list[str], cwd: Path | None = None, inp: bytes | None = None,
        check: bool = True, timeout: int | None = 1800) -> subprocess.CompletedProcess:
    p = subprocess.run(["git", *args], cwd=cwd, input=inp, capture_output=True,
                       timeout=timeout, env=GIT_ENV)
    if check and p.returncode != 0:
        msg = p.stderr.decode("utf-8", "replace").strip().replace("\n", " | ")
        raise GitError(f"git {' '.join(args[:2])} failed ({p.returncode}): {msg[:400]}")
    return p


def has_commit(repo_dir: Path, sha: str | None) -> bool:
    if not sha:
        return False
    return git(["cat-file", "-e", f"{sha}^{{commit}}"], cwd=repo_dir, check=False,
               timeout=120).returncode == 0


def rev_parse(repo_dir: Path, ref: str) -> str | None:
    p = git(["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], cwd=repo_dir,
            check=False, timeout=120)
    return p.stdout.decode().strip() or None if p.returncode == 0 else None


def chunk_by_chars(items: list[str], limit: int) -> list[list[str]]:
    out, cur, n = [], [], 0
    for it in items:
        if cur and n + len(it) + 1 > limit:
            out.append(cur)
            cur, n = [], 0
        cur.append(it)
        n += len(it) + 1
    if cur:
        out.append(cur)
    return out


# --------------------------------------------------------------------------- #
# GitHub API (one call per PR, cached, rate-limit aware)
# --------------------------------------------------------------------------- #

_pause_lock = threading.Lock()
_pause_until = 0.0


def _wait_if_paused() -> None:
    while True:
        with _pause_lock:
            wait = _pause_until - time.time()
        if wait <= 0:
            return
        time.sleep(min(wait, 30))


def _pause(seconds: float) -> None:
    global _pause_until
    with _pause_lock:
        target = time.time() + seconds
        if target > _pause_until:
            _pause_until = target
            print(f"[rate-limit] pausing API calls for {int(seconds)}s", flush=True)


def gh_get(url: str, token: str | None) -> tuple[int, dict | None]:
    headers = {"Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28",
               "User-Agent": "clones-before-code-research"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(6):
        _wait_if_paused()
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                remaining = r.headers.get("X-RateLimit-Remaining")
                reset = r.headers.get("X-RateLimit-Reset")
                if remaining is not None and reset and int(remaining) < 5:
                    _pause(max(5, int(reset) - time.time() + 5))
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                reset = e.headers.get("X-RateLimit-Reset")
                if e.headers.get("X-RateLimit-Remaining") == "0" and reset:
                    _pause(max(5, int(reset) - time.time() + 5))
                else:
                    ra = e.headers.get("Retry-After")
                    _pause(int(ra) if ra else 60 * (attempt + 1))
                continue
            if e.code >= 500:
                time.sleep(10 * (attempt + 1))
                continue
            return e.code, None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(10 * (attempt + 1))
    return -1, None


def api_pr(repo: str, n: int, token: str | None, api_dir: Path) -> tuple[int, dict | None]:
    cache = api_dir / safe(repo) / f"{n}.json"
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        return d["status"], d["data"]
    status, data = gh_get(f"https://api.github.com/repos/{repo}/pulls/{n}", token)
    if status in (200, 404, 410, 422):       # cache definitive answers only
        keep = None
        if data:
            keep = {k: data.get(k) for k in ("number", "state", "merged", "merged_at",
                                             "merge_commit_sha", "created_at", "closed_at",
                                             "commits", "changed_files", "additions",
                                             "deletions", "title")}
            keep["head"] = {"sha": data["head"]["sha"], "ref": data["head"]["ref"],
                            "repo": (data["head"].get("repo") or {}).get("full_name")}
            keep["base"] = {"sha": data["base"]["sha"], "ref": data["base"]["ref"],
                            "repo": (data["base"].get("repo") or {}).get("full_name")}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"status": status, "data": keep,
                                     "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                                    indent=1), encoding="utf-8")
        return status, keep
    return status, None


# --------------------------------------------------------------------------- #
# Git operations
# --------------------------------------------------------------------------- #


def ensure_clone(repo: str, dest: Path, timeout: int) -> str:
    if dest.exists():
        p = git(["rev-parse", "--is-bare-repository"], cwd=dest, check=False, timeout=60)
        if p.returncode == 0 and p.stdout.strip() == b"true":
            return "existing"
        shutil.rmtree(dest, ignore_errors=True)
    tmp = dest.with_name(dest.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    git(["-c", "core.longpaths=true", "clone", "--bare", "--filter=blob:none", "--quiet",
         f"https://github.com/{repo}.git", str(tmp)], timeout=timeout)
    tmp.rename(dest)
    return "cloned"


def fetch_refspecs(repo_dir: Path, refspecs: list[str]) -> list[str]:
    """Fetch in chunks; on chunk failure retry one by one. Returns failed refspecs."""
    failed: list[str] = []
    base = ["fetch", "--filter=blob:none", "--no-tags", "--quiet", "--no-write-fetch-head", "origin"]
    for ch in chunk_by_chars(refspecs, PATHSPEC_CHARS):
        if git([*base, *ch], cwd=repo_dir, check=False).returncode == 0:
            continue
        for rs in ch:
            if git([*base, rs], cwd=repo_dir, check=False).returncode != 0:
                failed.append(rs)
    return failed


def parse_raw_z(out: bytes) -> list[dict]:
    parts = out.split(b"\0")
    rows, i = [], 0
    while i < len(parts):
        meta = parts[i]
        if not meta.startswith(b":"):
            i += 1
            continue
        old_mode, new_mode, old_oid, new_oid, status = meta[1:].decode().split(" ")[:5]
        code = status[0]
        if code in "RC":
            old_path = parts[i + 1].decode("utf-8", "replace")
            new_path = parts[i + 2].decode("utf-8", "replace")
            i += 3
        else:
            old_path = new_path = parts[i + 1].decode("utf-8", "replace")
            i += 2
        rows.append({"status": code, "similarity": status[1:] or None,
                     "old_mode": old_mode, "new_mode": new_mode,
                     "old_oid": old_oid, "new_oid": new_oid,
                     "old_path": old_path, "new_path": new_path})
    return rows


def store_blobs(repo_dir: Path, oids: set[str], blob_dir: Path) -> tuple[int, list[str], list[str]]:
    need = sorted(o for o in oids if o != ZERO and not (blob_dir / o[:2] / o).exists())
    if not need:
        return 0, [], []
    p = git(["cat-file", "--batch"], cwd=repo_dir, inp=("\n".join(need) + "\n").encode(),
            check=False, timeout=3600)
    data, pos, got = p.stdout, 0, 0
    missing, too_large = [], []
    while pos < len(data):
        nl = data.find(b"\n", pos)
        if nl == -1:
            break
        header = data[pos:nl].decode("utf-8", "replace").split(" ")
        pos = nl + 1
        if len(header) < 3:                        # "<oid> missing" / "ambiguous"
            missing.append(header[0])
            continue
        oid, size = header[0], int(header[2])
        content = data[pos:pos + size]
        pos += size + 1
        if size > MAX_BLOB_BYTES:
            too_large.append(oid)
            continue
        dest = blob_dir / oid[:2] / oid
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(oid + ".tmp")
        tmp.write_bytes(content)
        tmp.replace(dest)
        got += 1
    return got, missing, too_large


def write_patch(repo_dir: Path, mb: str, head: str, paths: list[str], patch_path: Path) -> None:
    """Patch for selected paths; also makes git batch-prefetch the needed blobs."""
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    specs = [f":(literal){p}" for p in paths]
    with open(patch_path, "wb") as f:
        for ch in chunk_by_chars(specs, PATHSPEC_CHARS):
            p = git(["diff", "--no-color", "--no-ext-diff", "-M", "-U3", mb, head, "--", *ch],
                    cwd=repo_dir, check=False, timeout=3600)
            f.write(p.stdout)


# --------------------------------------------------------------------------- #
# Per-PR processing
# --------------------------------------------------------------------------- #


def classify(row: dict, spec_paths: set[str], langs: set[str]) -> tuple[str | None, str | None]:
    if row["old_mode"] == "160000" or row["new_mode"] == "160000":
        return None, None                                       # submodule pointer
    if row["new_path"] in spec_paths or row["old_path"] in spec_paths:
        return "spec", None
    path = row["new_path"] if row["status"] != "D" else row["old_path"]
    lang = language_of(path)
    if lang not in langs:
        return None, None
    pl = path.replace("\\", "/").lower()
    if EXCLUDE_RE.search(pl):
        return None, None
    return ("test" if TEST_RE.search(pl) else "code"), lang


def process_pr(ep: dict, repo_dir: Path, api: tuple[int, dict | None],
               fetch_failed: set[str], ctx: dict) -> dict:
    repo, n = ep["repo"], ep["pr_number"]
    rec = {"episode_id": f"{repo}#{n}", "repo": repo, "pr_number": n, "tool": ep["tool"],
           "languages": ";".join(ep["languages"]), "merged_at": ep["merged_at"],
           "base_branch": ep["base_branch"], "head_branch": ep["head_branch"],
           "api_status": api[0], "status": "ok", "error": None, "files": []}
    data = api[1] or {}
    api_head = (data.get("head") or {}).get("sha")
    api_base = (data.get("base") or {}).get("sha")
    rec.update(api_head_sha=api_head, api_base_sha=api_base,
               merge_commit_sha=data.get("merge_commit_sha"), api_commits=data.get("commits"),
               api_changed_files=data.get("changed_files"),
               head_repo=(data.get("head") or {}).get("repo"))

    ref_head = rev_parse(repo_dir, f"refs/pr/{n}/head")
    head = api_head if has_commit(repo_dir, api_head) else ref_head
    rec["head_sha"] = head
    rec["head_ref_mismatch"] = bool(api_head and ref_head and api_head != ref_head)
    if not head:
        rec["status"], rec["error"] = "no_head_commit", f"fetch failed: {sorted(fetch_failed)[:2]}"
        return rec

    mb, method = None, None
    if has_commit(repo_dir, api_base):
        p = git(["merge-base", api_base, head], cwd=repo_dir, check=False)
        if p.returncode == 0:
            mb, method = p.stdout.decode().strip(), "merge_base(api_base, head)"
    if not mb:
        base_ref = (data.get("base") or {}).get("ref") or ep["base_branch"]
        if base_ref and rev_parse(repo_dir, f"refs/heads/{base_ref}"):
            p = git(["merge-base", f"refs/heads/{base_ref}", head], cwd=repo_dir, check=False)
            if p.returncode == 0:
                mb, method = p.stdout.decode().strip(), "merge_base(current_base_branch, head)"
    rec["merge_base_sha"], rec["base_method"] = mb, method
    if not mb:
        rec["status"], rec["error"] = "no_merge_base", "base commit unavailable"
        return rec
    if mb == head:
        rec["status"], rec["error"] = "empty_range", "head is an ancestor of the base commit (base recorded after merge)"
        return rec

    raw = parse_raw_z(git(["diff", "--raw", "-z", "--no-abbrev", "-M", mb, head],
                          cwd=repo_dir).stdout)
    git_paths = {r["new_path"] for r in raw}
    sm_paths = ctx["pr_paths"].get((repo, n), set())
    union = git_paths | sm_paths
    rec["n_git_changed"] = len(git_paths)
    rec["n_specmine_changed"] = len(sm_paths)
    rec["path_jaccard_vs_specmine"] = round(len(git_paths & sm_paths) / len(union), 4) if union else None

    spec_paths = ctx["spec_paths"].get((repo, n), set())
    selected = []
    for r in raw:
        role, lang = classify(r, spec_paths, ctx["langs"])
        if role:
            selected.append({**r, "role": role, "lang": lang})
    rec["n_spec_files"] = sum(r["role"] == "spec" for r in selected)
    rec["n_code_files"] = sum(r["role"] == "code" for r in selected)
    rec["n_test_files"] = sum(r["role"] == "test" for r in selected)
    if rec["n_spec_files"] == 0 or rec["n_code_files"] == 0:
        rec["status"] = "no_spec_in_range" if rec["n_spec_files"] == 0 else "no_code_in_range"

    if selected:
        paths = sorted({p for r in selected for p in (r["old_path"], r["new_path"])})
        write_patch(repo_dir, mb, head, paths, ctx["patch_dir"] / f"{safe(repo)}__{n}.diff")
        oids = {o for r in selected for o in (r["old_oid"], r["new_oid"])}
        got, missing, too_large = store_blobs(repo_dir, oids, ctx["blob_dir"])
        rec["blobs_new"], rec["blobs_missing"], rec["blobs_too_large"] = got, len(missing), len(too_large)
        miss = set(missing) | set(too_large)
        for r in selected:
            r["old_blob_ok"] = r["old_oid"] == ZERO or r["old_oid"] not in miss
            r["new_blob_ok"] = r["new_oid"] == ZERO or r["new_oid"] not in miss
    rec["files"] = selected
    return rec


def process_repo(repo: str, eps: list[dict], ctx: dict) -> list[dict]:
    rec_dir: Path = ctx["rec_dir"]
    todo = [e for e in eps if not (rec_dir / f"{safe(repo)}__{e['pr_number']}.json").exists()]
    if not todo:
        return []
    out = []

    def save(rec: dict) -> None:
        path = rec_dir / f"{safe(repo)}__{rec['pr_number']}.json"
        path.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
        out.append(rec)

    repo_dir = ctx["repo_dir"] / f"{safe(repo)}.git"
    try:
        ensure_clone(repo, repo_dir, ctx["clone_timeout"])
    except (GitError, subprocess.TimeoutExpired) as e:
        for ep in todo:
            save({"episode_id": f"{repo}#{ep['pr_number']}", "repo": repo,
                  "pr_number": ep["pr_number"], "tool": ep["tool"],
                  "languages": ";".join(ep["languages"]), "status": "clone_failed",
                  "error": str(e)[:400], "files": []})
        return out

    apis = {ep["pr_number"]: api_pr(repo, ep["pr_number"], ctx["token"], ctx["api_dir"])
            for ep in todo}
    refspecs = [f"+refs/pull/{n}/head:refs/pr/{n}/head" for n in apis]
    for n, (_, d) in apis.items():
        base_sha = ((d or {}).get("base") or {}).get("sha")
        if base_sha and not has_commit(repo_dir, base_sha):
            refspecs.append(f"{base_sha}:refs/pr/{n}/base")
    failed = set(fetch_refspecs(repo_dir, refspecs))

    for ep in todo:
        try:
            rec = process_pr(ep, repo_dir, apis[ep["pr_number"]], failed, ctx)
        except (GitError, subprocess.TimeoutExpired) as e:
            rec = {"episode_id": f"{repo}#{ep['pr_number']}", "repo": repo,
                   "pr_number": ep["pr_number"], "tool": ep["tool"],
                   "languages": ";".join(ep["languages"]), "status": "git_error",
                   "error": str(e)[:400], "files": []}
        save(rec)
    return out


# --------------------------------------------------------------------------- #
# Inputs and summary
# --------------------------------------------------------------------------- #


def load_inputs(data_root: Path, out_dir: Path, langs: list[str]):
    ep_csv = out_dir / "10a_episode_languages.csv"
    if not ep_csv.exists():
        sys.exit(f"[error] {ep_csv} not found - run Step 10a first.")
    el = pd.read_csv(ep_csv)
    el = el[el["lang"].isin(langs)]
    el[["repo", "pr_number"]] = el["pr"].str.rsplit("#", n=1, expand=True)
    el["pr_number"] = el["pr_number"].astype(int)

    d = data_root / "data"
    prs = pd.read_parquet(d / "pull_requests.parquet",
                          columns=["tool", "repo_name", "pr_number", "merged_at",
                                   "base_branch", "head_branch"])
    files = pd.read_parquet(d / "pr_files.parquet",
                            columns=["repo_name", "pr_number", "file_path", "is_spec"])
    files["is_spec"] = as_flag(files["is_spec"])

    keys = set(zip(el["repo"], el["pr_number"]))
    files = files[[k in keys for k in zip(files["repo_name"], files["pr_number"])]]
    pr_paths, spec_paths = {}, {}
    for (repo, n), g in files.groupby(["repo_name", "pr_number"]):
        pr_paths[(repo, int(n))] = set(g["file_path"].astype(str))
        spec_paths[(repo, int(n))] = set(g.loc[g["is_spec"], "file_path"].astype(str))

    meta = prs.set_index(["repo_name", "pr_number"])
    episodes: dict[str, list[dict]] = {}
    for (repo, n), g in el.groupby(["repo", "pr_number"]):
        m = meta.loc[(repo, n)] if (repo, n) in meta.index else None
        if isinstance(m, pd.DataFrame):
            m = m.iloc[0]
        episodes.setdefault(repo, []).append({
            "repo": repo, "pr_number": int(n),
            "tool": g["tool_norm"].iloc[0],
            "languages": sorted(g["lang"].unique()),
            "merged_at": None if m is None else str(m["merged_at"]),
            "base_branch": None if m is None else m["base_branch"],
            "head_branch": None if m is None else m["head_branch"],
        })
    return episodes, pr_paths, spec_paths


def summarize(rec_dir: Path, out_dir: Path, n_scope_eps: int, n_scope_repos: int) -> None:
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(rec_dir.glob("*.json"))]
    if not recs:
        print("[summary] no records yet")
        return
    man = pd.DataFrame([{k: v for k, v in r.items() if k != "files"} for r in recs])
    rows = [{"episode_id": r["episode_id"], **f} for r in recs for f in r.get("files", [])]
    fil = pd.DataFrame(rows)
    man.to_csv(out_dir / "10b_episode_manifest.csv", index=False)
    fil.to_csv(out_dir / "10b_episode_files.csv", index=False)

    ok = man[man["status"] == "ok"]
    lines = [
        "STEP 10b - EPISODE ACQUISITION FUNNEL",
        "=" * 60,
        f"Episodes in scope (from 10a, selected languages) : {n_scope_eps:,}",
        f"Repositories in scope                            : {n_scope_repos:,}",
        f"Episodes processed so far                        : {len(man):,}",
        "",
        "STATUS",
        man["status"].value_counts().to_string(),
        "",
        "BASE METHOD (ok episodes)",
        ok["base_method"].value_counts(dropna=False).to_string() if len(ok) else "  none",
        "",
        "QUALITY FLAGS (ok episodes)",
    ]
    if len(ok):
        jac = pd.to_numeric(ok["path_jaccard_vs_specmine"], errors="coerce")
        lines += [
            f"  median path Jaccard, git range vs SpecMine pr_files : {jac.median():.3f}",
            f"  episodes with Jaccard < 0.8 (inspect)               : {(jac < 0.8).sum():,}",
            f"  head ref != API head sha                            : {int(ok['head_ref_mismatch'].sum()):,}",
            f"  episodes with missing/too-large blobs               : "
            f"{int(((ok['blobs_missing'].fillna(0) + ok['blobs_too_large'].fillna(0)) > 0).sum()):,}",
            "",
            "FILES IN OK EPISODES (by role and language)",
        ]
        okf = fil[fil["episode_id"].isin(ok["episode_id"])]
        if len(okf):
            lines.append(okf.groupby(["role", "lang"], dropna=False).size().to_string())
            lines += ["", "CHANGE STATUS OF SPEC FILES (A=added at this PR, M=modified, R=renamed, D=deleted)",
                      okf[okf["role"] == "spec"]["status"].value_counts().to_string()]
        lines += ["", "OK EPISODES BY LANGUAGE (an episode can have several)"]
        lines.append(ok["languages"].str.split(";").explode().value_counts().to_string())
    text = "\n".join(lines)
    (out_dir / "10b_summary.txt").write_text(text, encoding="utf-8")
    print(text)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="SpecMine release root")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--cache", default="cache/10b")
    ap.add_argument("--langs", default=",".join(LANGS_DEFAULT))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-repos", type=int, default=None, help="dry run on the first N repos")
    ap.add_argument("--repos", default=None, help="comma-separated owner/name list to process")
    ap.add_argument("--clone-timeout", type=int, default=3600)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    out_dir, cache = Path(args.out), Path(args.cache)
    dirs = {k: cache / k for k in ("repos", "blobs", "patches", "api", "records")}
    for p in [out_dir, *dirs.values()]:
        p.mkdir(parents=True, exist_ok=True)

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    episodes, pr_paths, spec_paths = load_inputs(Path(args.data), out_dir, langs)
    n_eps = sum(len(v) for v in episodes.values())

    if args.summarize_only:
        summarize(dirs["records"], out_dir, n_eps, len(episodes))
        return

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("[warn] GITHUB_TOKEN not set: unauthenticated API allows ~60 calls/hour. "
              "Fine for a 1-repo test, not for the full run.", flush=True)
    if git(["--version"], check=False).returncode != 0:
        sys.exit("[error] git not found on PATH")

    repos = sorted(episodes, key=lambda r: -len(episodes[r]))   # big repos first
    if args.repos:
        wanted = {r.strip() for r in args.repos.split(",")}
        repos = [r for r in repos if r in wanted]
    if args.max_repos:
        repos = sorted(repos, key=lambda r: len(episodes[r]))[: args.max_repos]  # small ones for dry run

    ctx = {"token": token, "langs": set(langs), "pr_paths": pr_paths, "spec_paths": spec_paths,
           "repo_dir": dirs["repos"], "blob_dir": dirs["blobs"], "patch_dir": dirs["patches"],
           "api_dir": dirs["api"], "rec_dir": dirs["records"], "clone_timeout": args.clone_timeout}

    print(f"[start] {len(repos)} repositories, "
          f"{sum(len(episodes[r]) for r in repos)} episodes, languages={langs}", flush=True)
    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_repo, r, episodes[r], ctx): r for r in repos}
        for fut in as_completed(futs):
            repo = futs[fut]
            done += 1
            try:
                recs = fut.result()
                ok = sum(r["status"] == "ok" for r in recs)
                print(f"[{done}/{len(repos)}] {repo}: {ok}/{len(recs)} ok  "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
            except Exception as e:                                # keep going
                print(f"[{done}/{len(repos)}] {repo}: FAILED {type(e).__name__}: {e}", flush=True)

    summarize(dirs["records"], out_dir, n_eps, len(episodes))


if __name__ == "__main__":
    main()