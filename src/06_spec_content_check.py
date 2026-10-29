from pathlib import Path
from urllib.parse import quote
import base64
import difflib
import os
import sys

import pandas as pd
import requests


RECONSTRUCTION_FILE = Path(
    "outputs/04_reconstruction_summary.csv"
)

TIMELINE_FILE = Path(
    "outputs/05_commit_timeline.csv"
)

BASE_FILE = Path(
    "outputs/05_spec_base_presence.csv"
)

OUTPUT_DIR = Path("outputs")

API_VERSION = "2026-03-10"


def github_session():

    session = requests.Session()

    session.headers.update(
        {
            "Accept":
                "application/vnd.github+json",

            "X-GitHub-Api-Version":
                API_VERSION,

            "User-Agent":
                "specmine-clones-before-code",
        }
    )

    token = os.environ.get(
        "GITHUB_TOKEN"
    )

    if token:
        session.headers.update(
            {
                "Authorization":
                    f"Bearer {token}"
            }
        )

    return session


def github_get(
    session,
    url,
    params=None,
    allow_404=False,
):

    response = session.get(
        url,
        params=params,
        timeout=30,
    )

    if (
        allow_404
        and response.status_code == 404
    ):
        return None

    if response.status_code != 200:

        raise RuntimeError(
            f"GitHub request failed\n"
            f"Status: {response.status_code}\n"
            f"URL: {response.url}\n"
            f"{response.text[:500]}"
        )

    return response.json()


def get_file_content(
    session,
    repo_name,
    path,
    ref,
):

    if not ref or pd.isna(ref):
        return None, "missing_ref"

    encoded_path = quote(
        str(path),
        safe="/",
    )

    url = (
        f"https://api.github.com/repos/"
        f"{repo_name}/contents/"
        f"{encoded_path}"
    )

    data = github_get(
        session,
        url,
        params={
            "ref": str(ref)
        },
        allow_404=True,
    )

    if data is None:
        return None, "not_found"

    if not isinstance(data, dict):
        return None, "unexpected_response"

    if data.get("type") != "file":
        return None, "not_a_file"

    encoded = data.get("content")

    if encoded:

        try:
            raw = base64.b64decode(
                encoded
            )

            text = raw.decode(
                "utf-8",
                errors="replace",
            )

            return text, "ok"

        except Exception:
            return None, "decode_error"

    # Large GitHub files can require blob retrieval.
    blob_sha = data.get("sha")

    if not blob_sha:
        return None, "content_missing"

    blob_url = (
        f"https://api.github.com/repos/"
        f"{repo_name}/git/blobs/"
        f"{blob_sha}"
    )

    blob = github_get(
        session,
        blob_url,
    )

    encoded = blob.get("content")

    if not encoded:
        return None, "blob_content_missing"

    raw = base64.b64decode(
        encoded
    )

    text = raw.decode(
        "utf-8",
        errors="replace",
    )

    return text, "ok_blob"


def split_paths(value):

    if pd.isna(value):
        return []

    return [
        item
        for item in str(value).split(";")
        if item
    ]


def first_touch_for_path(
    timeline,
    repo_name,
    pr_number,
    spec_path,
):

    rows = timeline[
        (timeline["repo_name"] == repo_name)
        &
        (
            timeline["pr_number"]
            == pr_number
        )
    ].sort_values("position")

    for _, row in rows.iterrows():

        touched = split_paths(
            row["spec_paths_touched"]
        )

        if spec_path in touched:

            return {
                "position":
                    int(row["position"]),

                "commit_sha":
                    row["commit_sha"],

                "commit_date":
                    row["commit_date"],

                "commit_message":
                    row["commit_message"],
            }

    return None


def make_diff(
    base_text,
    post_text,
    base_label,
    post_label,
):

    base_lines = (
        []
        if base_text is None
        else base_text.splitlines()
    )

    post_lines = (
        []
        if post_text is None
        else post_text.splitlines()
    )

    return list(
        difflib.unified_diff(
            base_lines,
            post_lines,
            fromfile=base_label,
            tofile=post_label,
            lineterm="",
            n=3,
        )
    )


def count_changes(diff_lines):

    added = 0
    removed = 0

    for line in diff_lines:

        if line.startswith("+++"):
            continue

        if line.startswith("---"):
            continue

        if line.startswith("+"):
            added += 1

        elif line.startswith("-"):
            removed += 1

    return added, removed


def main():

    required = [
        RECONSTRUCTION_FILE,
        TIMELINE_FILE,
        BASE_FILE,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:

        print("ERROR: Missing required files:")

        for path in missing:
            print(path)

        sys.exit(1)

    reconstruction = pd.read_csv(
        RECONSTRUCTION_FILE
    )

    timeline = pd.read_csv(
        TIMELINE_FILE
    )

    base_presence = pd.read_csv(
        BASE_FILE
    )

    session = github_session()

    output_rows = []
    diff_report = []

    for _, spec in (
        base_presence.iterrows()
    ):

        repo_name = spec["repo_name"]

        pr_number = int(
            spec["pr_number"]
        )

        spec_path = spec["spec_path"]

        base_lookup_path = (
            spec["base_lookup_path"]
        )

        exists_at_base = int(
            spec["exists_at_base"]
        )

        episode = reconstruction[
            (
                reconstruction["repo_name"]
                == repo_name
            )
            &
            (
                reconstruction["pr_number"]
                == pr_number
            )
        ]

        if len(episode) != 1:

            raise RuntimeError(
                f"Could not identify episode: "
                f"{repo_name} #{pr_number}"
            )

        episode = episode.iloc[0]

        base_sha = episode["base_sha"]
        head_sha = episode["head_sha"]

        first_touch = first_touch_for_path(
            timeline,
            repo_name,
            pr_number,
            spec_path,
        )

        # -------------------------------------------------
        # Retrieve pre-PR content.
        # -------------------------------------------------

        base_text = None
        base_status = "not_expected"

        if exists_at_base == 1:

            base_text, base_status = (
                get_file_content(
                    session,
                    repo_name,
                    base_lookup_path,
                    base_sha,
                )
            )

        # -------------------------------------------------
        # Retrieve content immediately after the first
        # PR commit that touched this spec path.
        # -------------------------------------------------

        if first_touch:

            post_ref = (
                first_touch[
                    "commit_sha"
                ]
            )

            post_text, post_status = (
                get_file_content(
                    session,
                    repo_name,
                    spec_path,
                    post_ref,
                )
            )

        else:

            post_ref = None
            post_text = None
            post_status = (
                "no_pr_spec_touch_found"
            )

        # -------------------------------------------------
        # Diff.
        # -------------------------------------------------

        diff_lines = make_diff(
            base_text,
            post_text,

            (
                f"{base_lookup_path}"
                f"@BASE"
            ),

            (
                f"{spec_path}"
                f"@FIRST_TOUCH"
            ),
        )

        added_lines, removed_lines = (
            count_changes(
                diff_lines
            )
        )

        # -------------------------------------------------
        # Simple automated temporal signal.
        #
        # NOT a requirement-level temporal label.
        # -------------------------------------------------

        if first_touch is None:

            content_signal = (
                "unresolved_no_pr_spec_commit"
            )

        elif exists_at_base == 0:

            content_signal = (
                "spec_file_introduced_in_pr"
            )

        elif (
            base_status.startswith("ok")
            and post_status.startswith("ok")
            and (
                added_lines > 0
                or removed_lines > 0
            )
        ):

            content_signal = (
                "preexisting_spec_modified_in_pr"
            )

        elif (
            base_status.startswith("ok")
            and post_status.startswith("ok")
        ):

            content_signal = (
                "preexisting_spec_no_"
                "text_change_detected"
            )

        else:

            content_signal = (
                "content_retrieval_incomplete"
            )

        output_rows.append(
            {
                "repo_name":
                    repo_name,

                "pr_number":
                    pr_number,

                "spec_path":
                    spec_path,

                "base_lookup_path":
                    base_lookup_path,

                "exists_at_base":
                    exists_at_base,

                "base_sha":
                    base_sha,

                "head_sha":
                    head_sha,

                "first_touch_position":
                    (
                        first_touch[
                            "position"
                        ]
                        if first_touch
                        else None
                    ),

                "first_touch_sha":
                    (
                        first_touch[
                            "commit_sha"
                        ]
                        if first_touch
                        else None
                    ),

                "first_touch_date":
                    (
                        first_touch[
                            "commit_date"
                        ]
                        if first_touch
                        else None
                    ),

                "base_fetch_status":
                    base_status,

                "post_fetch_status":
                    post_status,

                "base_characters":
                    (
                        len(base_text)
                        if base_text
                        else 0
                    ),

                "post_characters":
                    (
                        len(post_text)
                        if post_text
                        else 0
                    ),

                "added_lines":
                    added_lines,

                "removed_lines":
                    removed_lines,

                "content_signal":
                    content_signal,
            }
        )

        # -------------------------------------------------
        # Human-readable diff report.
        # -------------------------------------------------

        diff_report.append(
            "=" * 90
        )

        diff_report.append(
            f"{repo_name} PR #{pr_number}"
        )

        diff_report.append(
            f"SPEC: {spec_path}"
        )

        diff_report.append(
            f"EXISTED AT BASE: "
            f"{exists_at_base}"
        )

        diff_report.append(
            f"FIRST TOUCH: "
            f"{post_ref}"
        )

        diff_report.append(
            f"SIGNAL: "
            f"{content_signal}"
        )

        if first_touch:

            message = str(
                first_touch[
                    "commit_message"
                ]
            )

            diff_report.append(
                "COMMIT MESSAGE:"
            )

            diff_report.append(
                message
            )

        diff_report.append("")
        diff_report.append("DIFF:")
        diff_report.append("")

        if diff_lines:

            diff_report.extend(
                diff_lines
            )

        else:

            diff_report.append(
                "[No textual diff available]"
            )

        diff_report.append("")

    results = pd.DataFrame(
        output_rows
    )

    summary_csv = (
        OUTPUT_DIR /
        "06_spec_content_summary.csv"
    )

    summary_txt = (
        OUTPUT_DIR /
        "06_spec_content_summary.txt"
    )

    diff_txt = (
        OUTPUT_DIR /
        "06_spec_content_diffs.txt"
    )

    results.to_csv(
        summary_csv,
        index=False,
    )

    report = []

    report.append(
        "SPECMINE SPEC CONTENT "
        "TEMPORAL CHECK"
    )

    report.append("=" * 80)

    report.append("")
    report.append(
        "These are file/content signals, "
        "NOT final Tier A/B/C labels."
    )

    report.append("")
    report.append(
        results.to_string(
            index=False
        )
    )

    report.append("")
    report.append(
        "SIGNAL COUNTS"
    )

    report.append("-" * 80)

    report.append(
        results[
            "content_signal"
        ].value_counts().to_string()
    )

    summary_txt.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    diff_txt.write_text(
        "\n".join(diff_report),
        encoding="utf-8",
    )

    print()
    print(
        "Specification content check complete."
    )

    print()

    print(
        results[
            "content_signal"
        ].value_counts()
    )

    print()
    print("Created:")
    print(summary_csv)
    print(summary_txt)
    print(diff_txt)


if __name__ == "__main__":
    main()