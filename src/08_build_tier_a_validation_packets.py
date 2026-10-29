from pathlib import Path
import json
import sys
import difflib

import pandas as pd


SCREEN_FILE = Path(
    "outputs/07_temporal_candidates_prioritized.csv"
)

CACHE_DIR = Path(
    "outputs/github_cache_07"
)

OUTPUT_DIR = Path(
    "outputs/08_tier_a_packets"
)


def names_from_file(file_data):

    names = set()

    filename = file_data.get(
        "filename"
    )

    previous = file_data.get(
        "previous_filename"
    )

    if filename:
        names.add(filename)

    if previous:
        names.add(previous)

    return names


def get_commit_item(
    cached,
    position,
):

    details = cached.get(
        "commit_details",
        []
    )

    if (
        position < 1
        or position > len(details)
    ):
        return None

    return details[
        position - 1
    ]


def commit_message(detail):

    if not detail:
        return ""

    commit = detail.get(
        "commit",
        {}
    )

    return commit.get(
        "message",
        ""
    )


def classify_files(
    detail,
    spec_paths,
    code_paths,
):

    touched_specs = []
    touched_code = []
    other_files = []

    if not detail:
        return (
            touched_specs,
            touched_code,
            other_files,
        )

    for file_data in detail.get(
        "files",
        []
    ):

        filename = file_data.get(
            "filename"
        )

        names = names_from_file(
            file_data
        )

        if names.intersection(
            spec_paths
        ):

            touched_specs.append(
                file_data
            )

        elif names.intersection(
            code_paths
        ):

            touched_code.append(
                file_data
            )

        else:

            other_files.append(
                file_data
            )

    return (
        touched_specs,
        touched_code,
        other_files,
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

    patch = file_data.get(
        "patch"
    )

    lines = []

    lines.append(
        f"FILE: {filename}"
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

    lines.append("")

    if patch:

        lines.append("PATCH:")
        lines.append(patch)

    else:

        lines.append(
            "[GitHub patch unavailable]"
        )

    return "\n".join(lines)


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

        print(
            "Step 7 cache directory "
            "was not found."
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
        f"Tier-A screening candidates: "
        f"{len(candidates)}"
    )

    index_rows = []

    for number, (_, row) in enumerate(
        candidates.iterrows(),
        start=1,
    ):

        repo = row["repo_name"]

        pr = int(
            row["pr_number"]
        )

        tool = row["tool"]

        language = row[
            "repo_language"
        ]

        first_spec = int(
            row["first_spec_commit"]
        )

        first_code = int(
            row["first_code_commit"]
        )

        cache_name = (
            repo.replace("/", "__")
            + f"__pr_{pr}.json"
        )

        cache_path = (
            CACHE_DIR /
            cache_name
        )

        if not cache_path.exists():

            print(
                f"WARNING: cache missing "
                f"for {repo} #{pr}"
            )

            continue

        cached = json.loads(
            cache_path.read_text(
                encoding="utf-8"
            )
        )

        local_pr_files = (
            cached.get(
                "pr_files",
                []
            )
        )

        spec_paths = set()

        code_paths = set()

        # -----------------------------------------
        # Determine spec/code paths using the
        # screening information from commit details.
        #
        # Spec paths are identified from the
        # Step-7 commit timeline logic.
        # -----------------------------------------

        for item in cached.get(
            "commit_details",
            []
        ):

            detail = item.get(
                "detail"
            )

            if not detail:
                continue

            for file_data in detail.get(
                "files",
                []
            ):

                filename = file_data.get(
                    "filename",
                    ""
                )

                # Basic specification path markers
                # used by the selected tools.
                lower = filename.lower()

                if (
                    "/spec" in lower
                    or lower.startswith(
                        "specs/"
                    )
                    or lower.startswith(
                        "openspec/"
                    )
                    or "/requirements" in lower
                    or "/design" in lower
                    or "/tasks" in lower
                ):

                    spec_paths.add(
                        filename
                    )

        # -----------------------------------------
        # We already know which commits represent
        # first spec/code touches from Step 7.
        # Use those commit details directly.
        # -----------------------------------------

        spec_item = get_commit_item(
            cached,
            first_spec,
        )

        code_item = get_commit_item(
            cached,
            first_code,
        )

        spec_detail = (
            spec_item.get("detail")
            if spec_item
            else None
        )

        code_detail = (
            code_item.get("detail")
            if code_item
            else None
        )

        # All files from first code commit that
        # are not obvious specification/docs files
        # become implementation candidates.
        if code_detail:

            for file_data in (
                code_detail.get(
                    "files",
                    []
                )
            ):

                filename = file_data.get(
                    "filename",
                    ""
                )

                lower = filename.lower()

                if filename not in spec_paths:

                    if not (
                        lower.endswith(".md")
                        or lower.endswith(".txt")
                    ):

                        code_paths.add(
                            filename
                        )

        report = []

        report.append(
            "=" * 100
        )

        report.append(
            "TIER-A VALIDATION PACKET"
        )

        report.append(
            "=" * 100
        )

        report.append("")

        report.append(
            f"Candidate: {number}/"
            f"{len(candidates)}"
        )

        report.append(
            f"Repository: {repo}"
        )

        report.append(
            f"PR: #{pr}"
        )

        report.append(
            f"Tool: {tool}"
        )

        report.append(
            f"Language: {language}"
        )

        report.append("")

        report.append(
            f"First specification commit "
            f"position: {first_spec}"
        )

        report.append(
            f"First implementation commit "
            f"position: {first_code}"
        )

        report.append("")

        # =========================================
        # SPEC COMMIT
        # =========================================

        report.append(
            "=" * 100
        )

        report.append(
            "A. EARLIER SPECIFICATION COMMIT"
        )

        report.append(
            "=" * 100
        )

        if spec_item:

            report.append(
                f"SHA: "
                f"{spec_item.get('sha')}"
            )

        report.append("")

        report.append(
            "COMMIT MESSAGE:"
        )

        report.append(
            commit_message(
                spec_detail
            )
        )

        report.append("")

        if spec_detail:

            spec_files = []

            for file_data in (
                spec_detail.get(
                    "files",
                    []
                )
            ):

                filename = (
                    file_data.get(
                        "filename",
                        ""
                    )
                )

                lower = (
                    filename.lower()
                )

                if (
                    "/spec" in lower
                    or lower.startswith(
                        "specs/"
                    )
                    or lower.startswith(
                        "openspec/"
                    )
                    or "/requirements" in lower
                    or "/design" in lower
                    or "/tasks" in lower
                ):

                    spec_files.append(
                        file_data
                    )

            if spec_files:

                for file_data in (
                    spec_files
                ):

                    report.append(
                        format_patch(
                            file_data
                        )
                    )

                    report.append("")
                    report.append(
                        "-" * 100
                    )

            else:

                report.append(
                    "[No specification patch "
                    "identified automatically]"
                )

        # =========================================
        # COMMITS BETWEEN SPEC AND CODE
        # =========================================

        report.append("")

        report.append(
            "=" * 100
        )

        report.append(
            "B. COMMITS BETWEEN SPEC "
            "AND FIRST CODE CHANGE"
        )

        report.append(
            "=" * 100
        )

        if (
            first_code
            - first_spec
            <= 1
        ):

            report.append(
                "[No commits in between]"
            )

        else:

            for position in range(
                first_spec + 1,
                first_code,
            ):

                item = get_commit_item(
                    cached,
                    position,
                )

                if not item:
                    continue

                detail = item.get(
                    "detail"
                )

                report.append(
                    f"POSITION {position}"
                )

                report.append(
                    f"SHA: {item.get('sha')}"
                )

                report.append(
                    commit_message(
                        detail
                    )
                )

                report.append(
                    "-" * 100
                )

        # =========================================
        # CODE COMMIT
        # =========================================

        report.append("")

        report.append(
            "=" * 100
        )

        report.append(
            "C. FIRST IMPLEMENTATION COMMIT"
        )

        report.append(
            "=" * 100
        )

        if code_item:

            report.append(
                f"SHA: "
                f"{code_item.get('sha')}"
            )

        report.append("")

        report.append(
            "COMMIT MESSAGE:"
        )

        report.append(
            commit_message(
                code_detail
            )
        )

        report.append("")

        report.append(
            "IMPLEMENTATION FILES:"
        )

        if code_detail:

            found_code = False

            for file_data in (
                code_detail.get(
                    "files",
                    []
                )
            ):

                filename = (
                    file_data.get(
                        "filename",
                        ""
                    )
                )

                lower = filename.lower()

                if (
                    filename in spec_paths
                    or lower.endswith(".md")
                ):
                    continue

                found_code = True

                report.append(
                    format_patch(
                        file_data
                    )
                )

                report.append("")
                report.append(
                    "-" * 100
                )

            if not found_code:

                report.append(
                    "[No implementation patch "
                    "identified automatically]"
                )

        # =========================================
        # HUMAN VALIDATION FORM
        # =========================================

        report.append("")

        report.append(
            "=" * 100
        )

        report.append(
            "D. HUMAN VALIDATION"
        )

        report.append(
            "=" * 100
        )

        report.append(
            """
Answer these after reading sections A and C.

Q1. Does the earlier specification commit contain
a substantive behavioral requirement?

Answer:
    YES / NO / UNCERTAIN

Q2. Does the later implementation commit implement
that earlier requirement?

Answer:
    YES / NO / UNCERTAIN

Q3. Is the requirement already present BEFORE the
first relevant implementation change?

Answer:
    YES / NO / UNCERTAIN

Q4. Link confidence:

    HIGH
    MEDIUM
    LOW
    NONE

Q5. Preliminary temporal classification:

    TIER_A_CANDIDATE
    TIER_B
    TIER_C
    EXCLUDE
    UNCERTAIN

Reason:

Relevant requirement text:

Relevant implementation file/function:

Notes:
""".strip()
        )

        safe_repo = (
            repo.replace(
                "/",
                "__",
            )
        )

        packet_name = (
            f"{number:02}_"
            f"{safe_repo}_"
            f"pr_{pr}.txt"
        )

        packet_path = (
            OUTPUT_DIR /
            packet_name
        )

        packet_path.write_text(
            "\n".join(report),
            encoding="utf-8",
        )

        index_rows.append(
            {
                "packet":
                    packet_name,

                "tool":
                    tool,

                "repo_name":
                    repo,

                "pr_number":
                    pr,

                "language":
                    language,

                "first_spec_commit":
                    first_spec,

                "first_code_commit":
                    first_code,

                "spec_to_code_gap":
                    (
                        first_code
                        - first_spec
                    ),
            }
        )

        print(
            f"[{number:02}/"
            f"{len(candidates)}] "
            f"{repo} #{pr}"
        )

    index = pd.DataFrame(
        index_rows
    )

    index_path = (
        OUTPUT_DIR /
        "00_packet_index.csv"
    )

    index.to_csv(
        index_path,
        index=False,
    )

    print()
    print(
        "Validation packets created."
    )

    print(
        f"Folder: "
        f"{OUTPUT_DIR.resolve()}"
    )

    print(
        f"Index: "
        f"{index_path.resolve()}"
    )


if __name__ == "__main__":
    main()