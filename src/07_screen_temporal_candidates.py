from pathlib import Path
import json
import os
import sys
import time

import duckdb
import pandas as pd
import requests


API_VERSION = "2026-03-10"

PILOT_FILE = Path(
    "outputs/03_pilot_episodes.csv"
)

OUTPUT_DIR = Path("outputs")

CACHE_DIR = (
    OUTPUT_DIR /
    "github_cache_07"
)


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace(
        "'", "''"
    )


def github_session():

    token = os.environ.get(
        "GITHUB_TOKEN"
    )

    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN is not set."
        )

    session = requests.Session()

    session.headers.update(
        {
            "Accept":
                "application/vnd.github+json",

            "Authorization":
                f"Bearer {token}",

            "X-GitHub-Api-Version":
                API_VERSION,

            "User-Agent":
                "specmine-clones-before-code",
        }
    )

    return session


def get_json(
    session,
    url,
    params=None,
    allow_422=False,
):

    response = session.get(
        url,
        params=params,
        timeout=30,
    )

    if (
        allow_422
        and response.status_code == 422
    ):
        return None, "422_diff_unavailable"

    if response.status_code in {
        403, 429
    }:
        raise RuntimeError(
            "GitHub rate limit reached.\n"
            f"{response.text[:500]}"
        )

    if response.status_code != 200:
        return (
            None,
            f"http_{response.status_code}"
        )

    return response.json(), "ok"


def get_all_pages(
    session,
    url,
    allow_422=False,
):

    result = []
    page = 1

    while True:

        data, status = get_json(
            session,
            url,
            params={
                "per_page": 100,
                "page": page,
            },
            allow_422=allow_422,
        )

        if status != "ok":
            return [], status

        result.extend(data)

        if len(data) < 100:
            break

        page += 1

    return result, "ok"


def file_names(file_data):

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


def safe_cache_name(
    repo_name,
    pr_number,
):

    return (
        repo_name.replace("/", "__")
        + f"__pr_{pr_number}.json"
    )


def load_or_fetch_episode(
    session,
    repo_name,
    pr_number,
):

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = (
        CACHE_DIR /
        safe_cache_name(
            repo_name,
            pr_number,
        )
    )

    if cache_path.exists():

        data = json.loads(
            cache_path.read_text(
                encoding="utf-8"
            )
        )

        return data

    base_url = (
        f"https://api.github.com/repos/"
        f"{repo_name}"
    )

    pr_url = (
        f"{base_url}/pulls/"
        f"{pr_number}"
    )

    pr_data, pr_status = get_json(
        session,
        pr_url,
    )

    commits, commits_status = (
        get_all_pages(
            session,
            pr_url + "/commits",
        )
    )

    pr_files, files_status = (
        get_all_pages(
            session,
            pr_url + "/files",
            allow_422=True,
        )
    )

    commit_details = []

    if commits_status == "ok":

        for commit in commits:

            sha = commit.get("sha")

            detail, status = get_json(
                session,
                (
                    f"{base_url}/commits/"
                    f"{sha}"
                ),
            )

            commit_details.append(
                {
                    "sha": sha,
                    "status": status,
                    "detail": detail,
                }
            )

            # Keep the process serial and gentle.
            time.sleep(0.05)

    result = {
        "repo_name":
            repo_name,

        "pr_number":
            pr_number,

        "pr_status":
            pr_status,

        "pr":
            pr_data,

        "commits_status":
            commits_status,

        "commits":
            commits,

        "files_status":
            files_status,

        "pr_files":
            pr_files,

        "commit_details":
            commit_details,
    }

    cache_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return result


def classify_spec_statuses(
    spec_paths,
    pr_files,
):

    statuses = {}

    for spec_path in spec_paths:

        status = "unknown"

        for file_data in pr_files:

            if spec_path in file_names(
                file_data
            ):

                status = file_data.get(
                    "status",
                    "unknown",
                )

                break

        statuses[spec_path] = status

    return statuses


def main():

    if len(sys.argv) != 2:

        print("Usage:")
        print(
            r'python '
            r'src\07_screen_temporal_candidates.py '
            r'"D:\path\to\specmine-full"'
        )

        sys.exit(1)

    dataset_root = Path(
        sys.argv[1]
    ).resolve()

    pr_files_path = (
        dataset_root /
        "data" /
        "pr_files.parquet"
    )

    if not PILOT_FILE.exists():
        raise RuntimeError(
            f"Missing {PILOT_FILE}"
        )

    if not pr_files_path.exists():
        raise RuntimeError(
            f"Missing {pr_files_path}"
        )

    pilot = pd.read_csv(
        PILOT_FILE
    )

    con = duckdb.connect()

    con.execute(
        "SET memory_limit='8GB'"
    )

    con.execute(
        "SET threads=4"
    )

    con.execute(
        f"""
        CREATE VIEW pr_files AS
        SELECT *
        FROM read_parquet(
            '{sql_path(pr_files_path)}'
        )
        """
    )

    session = github_session()

    # Check authentication/rate limit first.
    rate, status = get_json(
        session,
        "https://api.github.com/rate_limit",
    )

    if status != "ok":
        raise RuntimeError(
            "Could not check GitHub "
            "rate limit."
        )

    core_rate = (
        rate["resources"]["core"]
    )

    print(
        "GitHub API limit:",
        core_rate["limit"],
    )

    print(
        "GitHub requests remaining:",
        core_rate["remaining"],
    )

    if core_rate["limit"] <= 60:
        raise RuntimeError(
            "GitHub token does not appear "
            "to be active."
        )

    summary_rows = []
    timeline_rows = []

    total = len(pilot)

    for counter, (_, episode) in enumerate(
        pilot.iterrows(),
        start=1,
    ):

        tool = episode["tool"]

        repo_name = (
            episode["repo_name"]
        )

        pr_number = int(
            episode["pr_number"]
        )

        print(
            f"[{counter:03}/{total}] "
            f"{repo_name} "
            f"#{pr_number}"
        )

        local = con.execute(
            """
            SELECT
                file_path,
                is_spec,
                is_code

            FROM pr_files

            WHERE tool = ?
              AND repo_name = ?
              AND pr_number = ?
            """,
            [
                tool,
                repo_name,
                pr_number,
            ],
        ).fetchdf()

        spec_paths = set(
            local.loc[
                local["is_spec"] == 1,
                "file_path",
            ].dropna()
        )

        code_paths = set(
            local.loc[
                local["is_code"] == 1,
                "file_path",
            ].dropna()
        )

        github = load_or_fetch_episode(
            session,
            repo_name,
            pr_number,
        )

        commits = github[
            "commit_details"
        ]

        pr_files = github[
            "pr_files"
        ]

        spec_statuses = (
            classify_spec_statuses(
                spec_paths,
                pr_files,
            )
        )

        added_specs = sum(
            status == "added"
            for status
            in spec_statuses.values()
        )

        preexisting_specs = sum(
            status in {
                "modified",
                "renamed",
                "removed",
            }
            for status
            in spec_statuses.values()
        )

        unknown_specs = sum(
            status == "unknown"
            for status
            in spec_statuses.values()
        )

        first_spec = None
        first_code = None

        for position, item in enumerate(
            commits,
            start=1,
        ):

            detail = item.get(
                "detail"
            )

            if not detail:
                continue

            touched = set()

            for file_data in detail.get(
                "files",
                []
            ):
                touched.update(
                    file_names(
                        file_data
                    )
                )

            touched_specs = sorted(
                spec_paths.intersection(
                    touched
                )
            )

            touched_code = sorted(
                code_paths.intersection(
                    touched
                )
            )

            if (
                touched_specs
                and first_spec is None
            ):
                first_spec = position

            if (
                touched_code
                and first_code is None
            ):
                first_code = position

            timeline_rows.append(
                {
                    "tool":
                        tool,

                    "repo_name":
                        repo_name,

                    "pr_number":
                        pr_number,

                    "position":
                        position,

                    "commit_sha":
                        item["sha"],

                    "touches_spec":
                        int(
                            bool(
                                touched_specs
                            )
                        ),

                    "touches_code":
                        int(
                            bool(
                                touched_code
                            )
                        ),

                    "spec_paths_touched":
                        ";".join(
                            touched_specs
                        ),

                    "code_paths_touched":
                        ";".join(
                            touched_code
                        ),
                }
            )

        # ---------------------------------------------
        # Screening classification.
        #
        # This is NOT the final Tier A/B/C label.
        # ---------------------------------------------

        if (
            github["commits_status"]
            != "ok"
            or len(commits) == 0
        ):

            screen_class = (
                "unresolved_history"
            )

        elif (
            first_spec is not None
            and first_code is not None
            and first_spec < first_code
        ):

            screen_class = (
                "spec_before_code_candidate"
            )

        elif preexisting_specs > 0:

            screen_class = (
                "preexisting_spec_"
                "content_check_needed"
            )

        elif (
            first_spec is not None
            and first_code is not None
            and first_spec == first_code
        ):

            screen_class = (
                "same_commit_new_spec_"
                "likely_tier_b"
            )

        elif (
            first_spec is not None
            and first_code is not None
            and first_code < first_spec
        ):

            screen_class = (
                "code_before_new_spec_"
                "not_tier_a"
            )

        else:

            screen_class = (
                "unresolved_touch_order"
            )

        summary_rows.append(
            {
                "tool":
                    tool,

                "repo_name":
                    repo_name,

                "pr_number":
                    pr_number,

                "repo_language":
                    episode[
                        "repo_language"
                    ],

                "github_pr_status":
                    github["pr_status"],

                "github_commits_status":
                    github[
                        "commits_status"
                    ],

                "github_files_status":
                    github[
                        "files_status"
                    ],

                "github_pr_commits":
                    len(commits),

                "local_spec_paths":
                    len(spec_paths),

                "local_code_paths":
                    len(code_paths),

                "spec_paths_added":
                    added_specs,

                "spec_paths_preexisting":
                    preexisting_specs,

                "spec_paths_unknown":
                    unknown_specs,

                "first_spec_commit":
                    first_spec,

                "first_code_commit":
                    first_code,

                "screen_class":
                    screen_class,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    timeline = pd.DataFrame(
        timeline_rows
    )

    priority_order = {
        "spec_before_code_candidate": 1,
        "preexisting_spec_content_check_needed": 2,
        "same_commit_new_spec_likely_tier_b": 3,
        "code_before_new_spec_not_tier_a": 4,
        "unresolved_touch_order": 5,
        "unresolved_history": 6,
    }

    summary[
        "priority"
    ] = (
        summary["screen_class"]
        .map(priority_order)
        .fillna(99)
    )

    prioritized = (
        summary.sort_values(
            [
                "priority",
                "tool",
                "repo_name",
                "pr_number",
            ]
        )
        .drop(
            columns=["priority"]
        )
    )

    summary_path = (
        OUTPUT_DIR /
        "07_temporal_screen.csv"
    )

    priority_path = (
        OUTPUT_DIR /
        "07_temporal_candidates_prioritized.csv"
    )

    timeline_path = (
        OUTPUT_DIR /
        "07_commit_timeline.csv"
    )

    report_path = (
        OUTPUT_DIR /
        "07_temporal_screen_summary.txt"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    prioritized.to_csv(
        priority_path,
        index=False,
    )

    timeline.to_csv(
        timeline_path,
        index=False,
    )

    counts = (
        summary[
            "screen_class"
        ]
        .value_counts()
    )

    report = []

    report.append(
        "SPECMINE 100-EPISODE "
        "TEMPORAL SCREEN"
    )

    report.append("=" * 80)

    report.append("")
    report.append(
        f"Episodes screened: "
        f"{len(summary)}"
    )

    report.append("")
    report.append(
        "SCREEN CLASSES"
    )

    report.append("-" * 80)

    report.append(
        counts.to_string()
    )

    report.append("")
    report.append(
        "BY TOOL AND SCREEN CLASS"
    )

    report.append("-" * 80)

    report.append(
        pd.crosstab(
            summary["tool"],
            summary["screen_class"],
        ).to_string()
    )

    report.append("")
    report.append(
        "BY LANGUAGE AND SCREEN CLASS"
    )

    report.append("-" * 80)

    report.append(
        pd.crosstab(
            summary[
                "repo_language"
            ],
            summary[
                "screen_class"
            ],
        ).to_string()
    )

    report.append("")
    report.append(
        "SPEC-BEFORE-CODE CANDIDATES"
    )

    report.append("-" * 80)

    candidates = summary[
        summary["screen_class"]
        == "spec_before_code_candidate"
    ]

    if len(candidates) == 0:

        report.append(
            "[NONE]"
        )

    else:

        report.append(
            candidates[
                [
                    "tool",
                    "repo_name",
                    "pr_number",
                    "repo_language",
                    "github_pr_commits",
                    "first_spec_commit",
                    "first_code_commit",
                ]
            ].to_string(
                index=False
            )
        )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print()
    print("=" * 80)

    print(
        "Temporal screening complete."
    )

    print("=" * 80)

    print()
    print(counts)

    print()
    print("Created:")
    print(summary_path)
    print(priority_path)
    print(timeline_path)
    print(report_path)


if __name__ == "__main__":
    main()