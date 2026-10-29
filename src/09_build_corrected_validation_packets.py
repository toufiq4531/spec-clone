from pathlib import Path
import json
import sys

import pandas as pd


SCREEN_FILE = Path(
    "outputs/07_temporal_candidates_prioritized.csv"
)

CACHE_DIR = Path(
    "outputs/github_cache_07"
)

OUTPUT_DIR = Path(
    "outputs/09_corrected_validation_packets"
)

MAX_PATCH_LINES = 120


def cache_name(repo_name, pr_number):

    return (
        repo_name.replace("/", "__")
        + f"__pr_{pr_number}.json"
    )


def get_commit_item(cached, position):

    commits = cached.get(
        "commit_details",
        []
    )

    if (
        position < 1
        or position > len(commits)
    ):
        return None

    return commits[position - 1]


def commit_message(detail):

    if not detail:
        return "[Commit detail unavailable]"

    return (
        detail.get(
            "commit",
            {}
        )
        .get(
            "message",
            "[No commit message]"
        )
    )


def format_patch(file_data):

    filename = file_data.get(
        "filename",
        "[unknown]"
    )

    status = file_data.get(
        "status",
        "unknown"
    )

    additions = file_data.get(
        "additions",
        0
    )

    deletions = file_data.get(
        "deletions",
        0
    )

    previous = file_data.get(
        "previous_filename"
    )

    patch = file_data.get(
        "patch"
    )

    lines = []

    lines.append(
        f"FILE: {filename}"
    )

    if previous:

        lines.append(
            f"PREVIOUS FILE: {previous}"
        )

    lines.append(
        f"STATUS: {status}"
    )

    lines.append(
        f"ADDITIONS: {additions}"
    )

    lines.append(
        f"DELETIONS: {deletions}"
    )

    if not patch:

        lines.append("")
        lines.append(
            "[GitHub patch unavailable]"
        )

        return "\n".join(lines)

    patch_lines = (
        patch.splitlines()
    )

    lines.append("")
    lines.append("PATCH:")

    if len(patch_lines) <= MAX_PATCH_LINES:

        lines.extend(
            patch_lines
        )

    else:

        lines.extend(
            patch_lines[
                :MAX_PATCH_LINES
            ]
        )

        lines.append("")
        lines.append(
            f"[PATCH TRUNCATED: "
            f"{len(patch_lines)} total lines]"
        )

    return "\n".join(lines)


def append_commit(
    report,
    item,
    position,
    screen_code_hint=None,
):

    report.append(
        "=" * 100
    )

    title = (
        f"COMMIT POSITION {position}"
    )

    if (
        screen_code_hint is not None
        and position == screen_code_hint
    ):

        title += (
            "  [STEP-7 AUTOMATED "
            "CODE HINT — NOT GROUND TRUTH]"
        )

    report.append(title)

    report.append(
        "=" * 100
    )

    if not item:

        report.append(
            "[Commit unavailable]"
        )

        report.append("")
        return

    report.append(
        f"SHA: {item.get('sha')}"
    )

    status = item.get(
        "status"
    )

    report.append(
        f"FETCH STATUS: {status}"
    )

    detail = item.get(
        "detail"
    )

    report.append("")

    report.append(
        "COMMIT MESSAGE:"
    )

    report.append(
        commit_message(
            detail
        )
    )

    report.append("")

    if not detail:

        report.append(
            "[No commit details]"
        )

        report.append("")
        return

    files = detail.get(
        "files",
        []
    )

    report.append(
        f"CHANGED FILES: {len(files)}"
    )

    report.append("")

    if len(files) == 0:

        report.append(
            "[No changed files returned]"
        )

    for number, file_data in enumerate(
        files,
        start=1,
    ):

        report.append(
            f"--- FILE {number}/{len(files)} ---"
        )

        report.append(
            format_patch(
                file_data
            )
        )

        report.append("")


def main():

    if not SCREEN_FILE.exists():

        print(
            f"ERROR: Missing {SCREEN_FILE}"
        )

        sys.exit(1)

    if not CACHE_DIR.exists():

        print(
            f"ERROR: Missing {CACHE_DIR}"
        )

        sys.exit(1)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    screen = pd.read_csv(
        SCREEN_FILE
    )

    candidates = screen[
        screen["screen_class"]
        == "spec_before_code_candidate"
    ].copy()

    print(
        "Candidates:",
        len(candidates)
    )

    index_rows = []
    annotation_rows = []

    for number, (_, row) in enumerate(
        candidates.iterrows(),
        start=1,
    ):

        repo_name = row[
            "repo_name"
        ]

        pr_number = int(
            row["pr_number"]
        )

        tool = row["tool"]

        language = row[
            "repo_language"
        ]

        first_spec = int(
            row["first_spec_commit"]
        )

        screen_code_hint = int(
            row["first_code_commit"]
        )

        path = (
            CACHE_DIR /
            cache_name(
                repo_name,
                pr_number,
            )
        )

        if not path.exists():

            print(
                "WARNING: missing cache:",
                repo_name,
                pr_number,
            )

            continue

        cached = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        commits = cached.get(
            "commit_details",
            []
        )

        total_commits = len(
            commits
        )

        report = []

        report.append(
            "=" * 100
        )

        report.append(
            "CORRECTED TIER-A "
            "VALIDATION PACKET"
        )

        report.append(
            "=" * 100
        )

        report.append("")

        report.append(
            f"Candidate: "
            f"{number}/{len(candidates)}"
        )

        report.append(
            f"Repository: {repo_name}"
        )

        report.append(
            f"PR: #{pr_number}"
        )

        report.append(
            f"Tool: {tool}"
        )

        report.append(
            f"Language: {language}"
        )

        report.append(
            f"Total PR commits: "
            f"{total_commits}"
        )

        report.append("")

        report.append(
            f"Earlier specification "
            f"commit position: "
            f"{first_spec}"
        )

        report.append(
            f"Step-7 automated "
            f"'first code' hint: "
            f"{screen_code_hint}"
        )

        report.append("")

        report.append(
            "IMPORTANT:"
        )

        report.append(
            "The Step-7 code position is only "
            "a screening hint."
        )

        report.append(
            "Your task is to identify the FIRST "
            "LATER COMMIT that actually implements "
            "the earlier requirement."
        )

        report.append("")

        # ==========================================
        # Earlier specification commit
        # ==========================================

        report.append(
            "#" * 100
        )

        report.append(
            "SECTION A — EARLIER "
            "SPECIFICATION COMMIT"
        )

        report.append(
            "#" * 100
        )

        report.append("")

        spec_item = get_commit_item(
            cached,
            first_spec,
        )

        append_commit(
            report,
            spec_item,
            first_spec,
        )

        # ==========================================
        # All later commits
        # ==========================================

        report.append("")
        report.append(
            "#" * 100
        )

        report.append(
            "SECTION B — ALL LATER "
            "COMMITS IN THIS PR"
        )

        report.append(
            "#" * 100
        )

        report.append("")

        if first_spec >= total_commits:

            report.append(
                "[There are no later commits]"
            )

        else:

            for position in range(
                first_spec + 1,
                total_commits + 1,
            ):

                item = get_commit_item(
                    cached,
                    position,
                )

                append_commit(
                    report,
                    item,
                    position,
                    screen_code_hint,
                )

                report.append("")

        # ==========================================
        # Annotation form
        # ==========================================

        report.append(
            "#" * 100
        )

        report.append(
            "SECTION C — HUMAN VALIDATION"
        )

        report.append(
            "#" * 100
        )

        report.append("")

        report.append(
            """
Q1. Does the earlier specification commit
INTRODUCE or substantively MODIFY a
behavioral requirement?

YES / NO / UNCERTAIN


Q2. Is there a LATER commit in this PR that
implements some or all of THAT requirement?

YES / NO / UNCERTAIN


Q3. What is the FIRST relevant implementation
commit?

Position:

SHA:

Commit message:

Relevant file(s):


Q4. Was the requirement present BEFORE that
first relevant implementation change?

YES / NO / UNCERTAIN


Q5. Requirement → implementation link confidence:

HIGH / MEDIUM / LOW / NONE


Q6. Preliminary classification:

TIER_A_CANDIDATE
TIER_B
TIER_C
EXCLUDE
UNCERTAIN


Reason:


Relevant requirement text:


Relevant implementation evidence:


Notes:
""".strip()
        )

        safe_repo = (
            repo_name.replace(
                "/",
                "__",
            )
        )

        filename = (
            f"{number:02}_"
            f"{safe_repo}_"
            f"pr_{pr_number}.txt"
        )

        output_path = (
            OUTPUT_DIR /
            filename
        )

        output_path.write_text(
            "\n".join(report),
            encoding="utf-8",
        )

        index_rows.append(
            {
                "candidate_number":
                    number,

                "packet":
                    filename,

                "tool":
                    tool,

                "repo_name":
                    repo_name,

                "pr_number":
                    pr_number,

                "language":
                    language,

                "total_pr_commits":
                    total_commits,

                "spec_commit_position":
                    first_spec,

                "step7_code_hint":
                    screen_code_hint,
            }
        )

        annotation_rows.append(
            {
                "candidate_number":
                    number,

                "repo_name":
                    repo_name,

                "pr_number":
                    pr_number,

                "tool":
                    tool,

                "language":
                    language,

                "q1_substantive_requirement":
                    "",

                "q2_later_implementation":
                    "",

                "first_relevant_impl_position":
                    "",

                "first_relevant_impl_sha":
                    "",

                "q4_requirement_before_impl":
                    "",

                "link_confidence":
                    "",

                "classification":
                    "",

                "relevant_requirement":
                    "",

                "implementation_evidence":
                    "",

                "reason_notes":
                    "",
            }
        )

        print(
            f"[{number:02}/"
            f"{len(candidates)}] "
            f"{repo_name} "
            f"#{pr_number}"
        )

    index_df = pd.DataFrame(
        index_rows
    )

    annotations_df = pd.DataFrame(
        annotation_rows
    )

    index_path = (
        OUTPUT_DIR /
        "00_corrected_packet_index.csv"
    )

    annotation_path = (
        OUTPUT_DIR /
        "00_annotation_template.csv"
    )

    index_df.to_csv(
        index_path,
        index=False,
    )

    annotations_df.to_csv(
        annotation_path,
        index=False,
    )

    print()
    print(
        "Corrected validation packets "
        "created."
    )

    print()
    print(
        f"Folder: {OUTPUT_DIR}"
    )

    print(
        f"Index: {index_path}"
    )

    print(
        f"Annotation template: "
        f"{annotation_path}"
    )


if __name__ == "__main__":
    main()