"""
Step 11 - Spec deltas -> requirement blocks, with template removal
===================================================================

Input : Step 10b outputs (manifest, files, cached blobs) for episodes with status ok.
Output: one row per requirement block INTRODUCED OR CHANGED by each episode.

Design (frozen choices, documented for the paper):
  * Document types. Primary, requirement-bearing documents:
        SpecKit  : specs/<feature>/spec.md                        -> sk_spec
        OpenSpec : openspec/changes/<id>/specs/**/spec.md         -> os_change_spec
                   openspec/specs/**/spec.md                      -> os_canonical_spec
    Secondary (implementation-prescriptive / planning) documents are segmented
    by section and kept for sensitivity analyses: plan, tasks, design, proposal,
    research, data-model, contracts, checklists, archive copies, other.
  * Blocks.
        OpenSpec : one "### Requirement:" with all of its scenarios; the delta
                   operation (ADDED / MODIFIED / REMOVED / RENAMED) is recorded.
        SpecKit  : one user story with its acceptance scenarios; each FR-/NFR-/
                   SC- list item; Edge Cases and Key Entities sections; other
                   sections as generic section blocks.
  * Delta. The head version and the base version of each file are segmented;
    a head block is part of the episode's delta if its normalized text does not
    occur among the base blocks (new or modified). Deleted files contribute none.
  * Template removal ("verified-template removal"). A line is removed if its
    normalized form occurs in ANY historical version of the tools' own templates
    (11_template_lines.json, built from github/spec-kit and Fission-AI/OpenSpec).
    Bold labels that occur in templates (e.g. **WHEN**, **Why this priority**)
    are stripped from filled lines. HTML comments and pure placeholders are dropped.
    Frequency is NOT used to remove text; frequent residual blocks are listed in
    the summary for manual review instead.
  * Two normalized texts per block:
        norm_text           primary exposure: fenced code, file paths, URLs removed
        norm_text_with_code sensitivity: code and paths kept
  * Tokens: lower-cased [a-z0-9]+ words; CJK/kana/hangul as single characters.
    Requirement IDs (FR-001, SC-002, T012, US1, REQ-3) are removed. Negations,
    numbers and units are kept.

Usage:
    python src\\11_spec_blocks.py              (run from the project root)
Needs: src\\11_template_lines.json next to this script.
Outputs:
    outputs\\11_spec_blocks.parquet
    outputs\\11_spec_blocks_summary.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ZERO = "0" * 40
MIN_SUBSTANTIVE_TOKENS = 8          # frozen before any similarity is computed

# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #

COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
FENCE_RE = re.compile(r"^\s*(```|~~~)")
MD_PREFIX_RE = re.compile(r"^\s*(#{1,6}\s+|>\s*|[-*+]\s+(\[[ xX]\]\s+)?|\d+[.)]\s+)+")
BOLD_RE = re.compile(r"\*\*([^*]{1,60}?)\*\*:?")
INLINE_CODE_RE = re.compile(r"`([^`]*)`")
URL_RE = re.compile(r"https?://\S+")
PATHLIKE_RE = re.compile(r"^[\w.@~-]*[/\\][\w./\\@~-]*$|^[\w-]+\.[a-z]{1,5}$", re.I)
REQ_ID_RE = re.compile(r"\b(?:FR|NFR|SC|REQ|US|T|AC|TC|CHK)-?\d{1,4}[a-z]?\b", re.I)
PLACEHOLDER_LINE_RE = re.compile(r"^\s*([-*]\s*)?\[[^\]]{1,120}\]\s*$")
TOKEN_RE = re.compile(r"[a-z0-9]+|[぀-ヿ㐀-䶿一-鿿가-힯]")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
LIST_ITEM_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+")
REQ_ITEM_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s+(\*\*)?\s*((?:FR|NFR|SC|REQ)-?\d{1,4}[a-z]?)", re.I)
LABEL_PREFIX_RE = re.compile(
    r"^(requirement|scenario|user story\s*\d*\s*[-:–]?)\s*:?\s*", re.I)
PRIORITY_RE = re.compile(r"[(\uff08]\s*priority\s*[:\uff1a]\s*p\d\s*[)\uff09]", re.I)


def line_key(line: str) -> str:
    """Normalization used to compare a document line with template lines."""
    s = COMMENT_RE.sub("", line)
    s = MD_PREFIX_RE.sub("", s)
    s = s.replace("**", "").replace("__", "").replace("`", "")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def load_templates(path: Path) -> tuple[set[str], set[str]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    keys, labels = set(), set()
    for lines in d["lines"].values():
        for ln in lines:
            k = line_key(ln)
            if k and len(TOKEN_RE.findall(k)) >= 1:
                keys.add(k)
            for m in BOLD_RE.finditer(ln):
                lab = m.group(1).strip().lower()
                if 0 < len(lab.split()) <= 5 and not REQ_ID_RE.fullmatch(lab):
                    labels.add(lab)
    return keys, labels


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(REQ_ID_RE.sub(" ", text.lower()))


def clean_block(lines: list[str], tpl_keys: set[str], tpl_labels: set[str]) -> dict:
    """Returns template-stripped normalized texts and bookkeeping counts."""
    kept_code, kept_primary = [], []
    removed = 0
    in_fence = False
    has_fence = False
    for raw in lines:
        if FENCE_RE.match(raw):
            in_fence = not in_fence
            has_fence = True
            continue
        if in_fence:
            kept_code.append(raw)               # code only in the sensitivity text
            continue
        line = COMMENT_RE.sub("", raw)
        if not line.strip():
            continue
        if PLACEHOLDER_LINE_RE.match(line) or "$ARGUMENTS" in line:
            removed += 1
            continue
        if line_key(line) in tpl_keys:
            removed += 1
            continue

        def strip_label(m: re.Match) -> str:
            return "" if m.group(1).strip().lower() in tpl_labels else m.group(1)
        line = BOLD_RE.sub(strip_label, line)
        h = HEADING_RE.match(line)
        if h:
            line = PRIORITY_RE.sub("", LABEL_PREFIX_RE.sub("", h.group(2)))
        line = MD_PREFIX_RE.sub("", line)
        prim = URL_RE.sub(" ", line)
        prim = INLINE_CODE_RE.sub(lambda m: " " if PATHLIKE_RE.match(m.group(1).strip()) else m.group(1), prim)
        kept_primary.append(prim)
        kept_code.append(INLINE_CODE_RE.sub(lambda m: m.group(1), line))
    t_prim = tokens(" ".join(kept_primary))
    t_code = tokens(" ".join(kept_code))
    return {"norm_text": " ".join(t_prim), "norm_text_with_code": " ".join(t_code),
            "n_tokens": len(t_prim), "n_tokens_with_code": len(t_code),
            "template_lines_removed": removed, "has_code_fence": has_fence}


# --------------------------------------------------------------------------- #
# Document typing
# --------------------------------------------------------------------------- #


def doc_type(path: str) -> tuple[str, str | None, str | None]:
    """Returns (doc_type, feature_key, capability)."""
    p = path.replace("\\", "/")
    low = p.lower()
    name = low.rsplit("/", 1)[-1]
    m = re.search(r"(?:^|/)openspec/changes/archive/([^/]+)/", low)
    if m:
        return ("os_archive_spec" if name == "spec.md" else "os_archive_other"), f"change:{m.group(1)}", None
    m = re.search(r"(?:^|/)openspec/changes/([^/]+)/(.*)$", p, re.I)
    if m:
        cid, rest = m.group(1), m.group(2)
        cap = re.search(r"specs/(.+)/spec\.md$", rest, re.I)
        if name == "spec.md" and cap:
            return "os_change_spec", f"change:{cid}", cap.group(1)
        kind = {"proposal.md": "os_proposal", "design.md": "os_design", "tasks.md": "os_tasks"}.get(name, "os_other")
        return kind, f"change:{cid}", None
    m = re.search(r"(?:^|/)openspec/specs/(.+)/spec\.md$", p, re.I)
    if m:
        return "os_canonical_spec", f"capability:{m.group(1)}", m.group(1)
    if "/openspec/" in "/" + low:
        return "os_other", None, None
    m = re.search(r"(?:^|/)specs?/([^/]+)/(.*)$", p, re.I)
    if m:
        feat, rest = m.group(1), m.group(2).lower()
        kind = {"spec.md": "sk_spec", "plan.md": "sk_plan", "tasks.md": "sk_tasks",
                "research.md": "sk_research", "data-model.md": "sk_data_model",
                "quickstart.md": "sk_quickstart"}.get(rest, None)
        if kind is None:
            kind = ("sk_contract" if rest.startswith("contracts/") else
                    "sk_checklist" if rest.startswith("checklists/") else "sk_other")
        return kind, f"feature:{feat}", None
    return "other", None, None


PRIMARY_DOCS = {"sk_spec", "os_change_spec", "os_canonical_spec"}

# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #


def _headings(lines: list[str]) -> list[tuple[int, int, str]]:
    out, fence = [], False
    for i, ln in enumerate(lines):
        if FENCE_RE.match(ln):
            fence = not fence
            continue
        if fence:
            continue
        m = HEADING_RE.match(ln)
        if m:
            out.append((i, len(m.group(1)), m.group(2).strip()))
    return out


def _section_end(heads, idx, level, n_lines) -> int:
    for j in range(idx + 1, len(heads)):
        if heads[j][1] <= level:
            return heads[j][0]
    return n_lines


def segment_openspec(lines: list[str]) -> list[dict]:
    heads = _headings(lines)
    blocks, op = [], None
    for k, (i, lvl, title) in enumerate(heads):
        t = title.lower()
        if lvl == 2:
            m = re.match(r"(added|modified|removed|renamed)\b", t)
            op = m.group(1).upper() if m else ("PURPOSE" if t.startswith("purpose") else None)
            if op == "PURPOSE":
                end = _section_end(heads, k, 2, len(lines))
                blocks.append({"kind": "os_purpose", "op": None, "title": title, "start": i, "end": end})
        elif lvl == 3 and t.startswith("requirement"):
            end = _section_end(heads, k, 3, len(lines))
            kind = {"REMOVED": "os_removed", "RENAMED": "os_renamed"}.get(op, "os_requirement")
            blocks.append({"kind": kind, "op": op, "title": title, "start": i, "end": end})
    if not blocks:                                   # non-standard layout: fall back to sections
        return segment_sections(lines)
    return blocks


def segment_speckit(lines: list[str]) -> list[dict]:
    heads = _headings(lines)
    blocks, covered = [], set()
    for k, (i, lvl, title) in enumerate(heads):
        t = title.lower()
        end = _section_end(heads, k, lvl, len(lines))
        if lvl >= 3 and re.match(r"user stor(y|ies)\b", t):
            blocks.append({"kind": "sk_user_story", "op": None, "title": title, "start": i, "end": end})
            covered.update(range(i, end))
        elif "edge case" in t:
            blocks.append({"kind": "sk_edge_cases", "op": None, "title": title, "start": i, "end": end})
            covered.update(range(i, end))
        elif "key entit" in t:
            blocks.append({"kind": "sk_entities", "op": None, "title": title, "start": i, "end": end})
            covered.update(range(i, end))
    # requirement list items anywhere (FR-, NFR-, SC-, REQ-)
    i = 0
    while i < len(lines):
        m = REQ_ITEM_RE.match(lines[i])
        if m and i not in covered:
            indent = len(LIST_ITEM_RE.match(lines[i]).group(1))
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if not nxt.strip() or HEADING_RE.match(nxt):
                    break
                li = LIST_ITEM_RE.match(nxt)
                if li and len(li.group(1)) <= indent:
                    break
                j += 1
            rid = m.group(3).upper()
            kind = "sk_sc" if rid.startswith("SC") else "sk_requirement"
            blocks.append({"kind": kind, "op": None, "title": rid, "start": i, "end": j})
            covered.update(range(i, j))
            i = j
        else:
            i += 1
    # remaining text by section (secondary)
    for b in segment_sections(lines):
        rest = [x for x in range(b["start"], b["end"]) if x not in covered]
        if rest:
            blocks.append({**b, "start": rest[0], "end": rest[-1] + 1, "lines_subset": rest})
    return blocks


def segment_sections(lines: list[str]) -> list[dict]:
    heads = [h for h in _headings(lines) if h[1] <= 3]
    if not heads:
        return [{"kind": "section", "op": None, "title": None, "start": 0, "end": len(lines)}]
    blocks = []
    if heads[0][0] > 0:
        blocks.append({"kind": "section", "op": None, "title": None, "start": 0, "end": heads[0][0]})
    for k, (i, _lvl, title) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        blocks.append({"kind": "section", "op": None, "title": title, "start": i, "end": end})
    return blocks


def segment(text: str, dtype: str) -> list[dict]:
    lines = text.split("\n")
    if dtype in ("os_change_spec", "os_canonical_spec", "os_archive_spec"):
        blocks = segment_openspec(lines)
    elif dtype == "sk_spec":
        blocks = segment_speckit(lines)
    else:
        blocks = segment_sections(lines)
    for b in blocks:
        idx = b.pop("lines_subset", range(b["start"], b["end"]))
        b["lines"] = [lines[x] for x in idx]
    return blocks


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def read_blob(blob_dir: Path, oid) -> str | None:
    if not isinstance(oid, str) or oid == ZERO:
        return None
    p = blob_dir / oid[:2] / oid
    if not p.exists():
        return None
    b = p.read_bytes()
    if b"\0" in b[:8000]:
        return None
    return b.decode("utf-8", "replace").replace("\r\n", "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--cache", default="cache/10b")
    ap.add_argument("--templates", default=str(Path(__file__).with_name("11_template_lines.json")))
    args = ap.parse_args()
    out, blob_dir = Path(args.out), Path(args.cache) / "blobs"

    tpl_keys, tpl_labels = load_templates(Path(args.templates))
    man = pd.read_csv(out / "10b_episode_manifest.csv")
    man = man[man["status"] == "ok"]
    files = pd.read_csv(out / "10b_episode_files.csv")
    files = files[(files["role"] == "spec") & files["episode_id"].isin(man["episode_id"])]
    tool_of = dict(zip(man["episode_id"], man["tool"]))
    repo_of = dict(zip(man["episode_id"], man["repo"]))

    archive_eps = set(files.loc[(files["status"] == "R") &
                                files["new_path"].str.contains("/changes/archive/", case=False, na=False),
                                "episode_id"])

    rows, file_stats = [], []
    for f in files.itertuples(index=False):
        path = f.new_path if f.status != "D" else f.old_path
        dtype, feature, cap = doc_type(path)
        stat = {"episode_id": f.episode_id, "path": path, "doc_type": dtype, "status": f.status,
                "outcome": "ok"}
        if not str(path).lower().endswith((".md", ".markdown")):
            stat["outcome"] = "not_markdown"
            file_stats.append(stat)
            continue
        if f.status == "D":
            stat["outcome"] = "deleted"
            file_stats.append(stat)
            continue
        head = read_blob(blob_dir, f.new_oid)
        if head is None:
            stat["outcome"] = "blob_unavailable"
            file_stats.append(stat)
            continue
        base = read_blob(blob_dir, f.old_oid) if f.status in ("M", "R") else None
        base_norms = set()
        base_titles = set()
        if base:
            for b in segment(base, doc_type(f.old_path)[0]):
                c = clean_block(b["lines"], tpl_keys, tpl_labels)
                base_norms.add(c["norm_text_with_code"])
                base_titles.add((b["kind"], (b["title"] or "").strip().lower()))
        n_kept = 0
        for idx, b in enumerate(segment(head, dtype)):
            c = clean_block(b["lines"], tpl_keys, tpl_labels)
            if c["norm_text_with_code"] in base_norms:
                continue                                   # unchanged block: not part of this delta
            delta = ("new" if f.status == "A" or
                     (b["kind"], (b["title"] or "").strip().lower()) not in base_titles else "modified")
            primary = dtype in PRIMARY_DOCS and b["kind"] in {
                "os_requirement", "sk_user_story", "sk_requirement", "sk_edge_cases"}
            lifecycle = None
            if dtype == "os_canonical_spec" and f.episode_id in archive_eps:
                lifecycle = "archive_sync"
            elif dtype in ("os_archive_spec", "os_archive_other"):
                lifecycle = "archive_copy"
            rows.append({
                "episode_id": f.episode_id, "repo": repo_of[f.episode_id], "tool": tool_of[f.episode_id],
                "file_path": path, "file_status": f.status, "doc_type": dtype,
                "feature_key": feature, "capability": cap,
                "block_idx": idx, "block_kind": b["kind"], "op": b["op"], "delta_type": delta,
                "title": b["title"], "start_line": b["start"] + 1, "end_line": b["end"],
                "raw_text": "\n".join(b["lines"]), **c,
                "norm_hash": hashlib.sha1(c["norm_text"].encode()).hexdigest()[:16],
                "primary": primary and lifecycle is None,
                "substantive": c["n_tokens"] >= MIN_SUBSTANTIVE_TOKENS,
                "lifecycle": lifecycle,
            })
            n_kept += 1
        stat["delta_blocks"] = n_kept
        file_stats.append(stat)

    blocks = pd.DataFrame(rows)
    if blocks.empty:
        print("[warn] no blocks produced")
        return
    # within-episode duplicates (same requirement written into several files of one PR)
    blocks = blocks.sort_values(["episode_id", "primary", "doc_type"], ascending=[True, False, True])
    blocks["dup_within_episode"] = blocks.duplicated(["episode_id", "norm_hash"]) & (blocks["n_tokens"] > 0)
    blocks.to_parquet(out / "11_spec_blocks.parquet", index=False)
    fs = pd.DataFrame(file_stats)
    fs.to_csv(out / "11_spec_file_outcomes.csv", index=False)

    # ---------------- summary ---------------- #
    P = blocks[blocks["primary"] & blocks["substantive"] & ~blocks["dup_within_episode"]]
    n_eps = man["episode_id"].nunique()
    eps_with = P["episode_id"].nunique()
    raw_lines = blocks["raw_text"].str.count("\n").add(1).sum()
    by_tool = (P.groupby("tool").agg(episodes=("episode_id", "nunique"), blocks=("norm_hash", "size"),
                                     median_tokens=("n_tokens", "median"))
               .join(man.groupby("tool")["episode_id"].nunique().rename("ok_episodes")))
    h = P.groupby("norm_hash").agg(n=("episode_id", "nunique"), repos=("repo", "nunique"),
                                   text=("norm_text", "first"))
    exact_multi = h[h["n"] > 1]
    top = h.sort_values(["repos", "n"], ascending=False).head(25)
    lines = [
        "STEP 11 - SPEC REQUIREMENT BLOCKS",
        "=" * 60,
        f"Template lines loaded: {len(tpl_keys):,} keys, {len(tpl_labels):,} bold labels",
        f"Minimum substantive tokens: {MIN_SUBSTANTIVE_TOKENS}",
        "",
        "SPEC FILE OUTCOMES",
        fs["outcome"].value_counts().to_string(),
        "",
        "DOCUMENT TYPES (spec files processed)",
        fs.loc[fs["outcome"] == "ok", "doc_type"].value_counts().to_string(),
        "",
        "DELTA BLOCKS BY KIND (all docs)",
        blocks.groupby(["block_kind", "primary"]).size().to_string(),
        "",
        f"Template lines removed: {int(blocks['template_lines_removed'].sum()):,} "
        f"of ~{int(raw_lines):,} raw lines ({blocks['template_lines_removed'].sum() / max(raw_lines, 1):.1%})",
        f"Lifecycle-flagged blocks: {blocks['lifecycle'].value_counts().to_dict()}",
        "",
        "PRIMARY SUBSTANTIVE BLOCKS (exposure population for RQ1/RQ2)",
        f"  ok episodes                          : {n_eps:,}",
        f"  episodes with >=1 primary block      : {eps_with:,} ({eps_with / n_eps:.1%})",
        f"  primary blocks                       : {len(P):,}",
        by_tool.to_string(),
        "",
        "EXACT-DUPLICATE PREVIEW (identical normalized primary blocks)",
        f"  distinct block texts                 : {len(h):,}",
        f"  texts appearing in >1 episode        : {len(exact_multi):,}",
        f"  ...of which in >1 repository         : {int((exact_multi['repos'] > 1).sum()):,}",
        "",
        "MOST WIDESPREAD PRIMARY BLOCK TEXTS - REVIEW FOR RESIDUAL BOILERPLATE",
        "(if these are template text, the template list is missing something)",
    ]
    for hsh, r in top.iterrows():
        lines.append(f"  [{r['repos']} repos, {r['n']} eps] {r['text'][:150]}")
    text = "\n".join(lines)
    (out / "11_spec_blocks_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()