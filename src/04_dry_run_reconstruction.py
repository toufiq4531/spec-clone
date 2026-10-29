from pathlib import Path
from datetime import datetime, timezone
import json
import os
import sys

import duckdb
import pandas as pd
import requests


# ---------------------------------------------------------
# Five deliberately different pilot episodes.
# ---------------------------------------------------------

DRY_RUN = [
    {
        "tool": "speckit",
        "repo_name": "timcsy/ai-api",
        "pr_number": 53,
        "case_type": "complete_history",
    },
    {
        "tool": "claude_code",
        "repo_name": "gaia-react/gaia",
        "pr_number": 606,
        "case_type": "complete_history_other_tool",
    },
    {
        "tool": "openspec",
        "repo_name": "Soju06/codex-lb",
        "pr_number": 137,
        "case_type": "partial_history",
    },
    {
        "tool": "openspec",
        "repo_name": "epam/ai-dial-admin-frontend",
        "pr_number": 3846,
        "case_type": "no_history",
    },
    {
        "tool": "openspec",
        "repo_name": "carverauto/serviceradar",
        "pr_number": 2855,
        "case_type": "truncated_tree_partial_history",
    },
]


API_VERSION = "2026-03-10"


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def github_session():
    session = requests.Session()

    session.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": "specmine-clones-before-code",
        }
    )

    # Optional. We do not require a token for this
    # five-episode public-repository dry run.
    token = os.environ.get("GITHUB_TOKEN")

    if token:
        session.headers.update(
            {
                "Authorization": f"Bearer {token}"
            }
        )

    return session


def github_get(session, url, params=None):

    response = session.get(
        url,
        params=params,
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"GitHub request failed: "
            f"{response.status_code} "
            f"{response.url}\n"
            f"{response.text[:500]}"
        )

    return response.json()


def github_get_all_pages(
    session,
    url,
):

    items = []
    page = 1

    while True:

        data = github_get(
            session,
            url,
            params={
                "per_page": 100,
                "page": page,
            },
        )

        if not isinstance(data, list):
            raise RuntimeError(
                f"Expected list response from "
                f"{url}"
            )

        items.extend(data)

        if len(data) < 100:
            break

        page += 1

    return items


def main():

    if len(sys.argv) != 2:
        print("Usage:")
        print(
            r'python src\04_dry_run_reconstruction.py '
            r'"D:\path\to\specmine-full"'
        )
        sys.exit(1)

    root = Path(sys.argv[1]).resolve()
    data_dir = root / "data"

    pilot_file = Path(
        "outputs/03_pilot_episodes.csv"
    )

    pr_files_path = (
        data_dir / "pr_files.parquet"
    )

    history_path = (
        data_dir /
        "spec_file_commits.parquet"
    )

    if not pilot_file.exists():
        print(
            "ERROR: Missing "
            "outputs/03_pilot_episodes.csv"
        )
        sys.exit(1)

    if not pr_files_path.exists():
        print(
            f"ERROR: Missing {pr_files_path}"
        )
        sys.exit(1)

    if not history_path.exists():
        print(
            f"ERROR: Missing {history_path}"
        )
        sys.exit(1)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    pilot = pd.read_csv(pilot_file)

    # ---------------------------------------------------------
    # Confirm that all five episodes really exist in the
    # selected 100-episode pilot.
    # ---------------------------------------------------------

    selected_rows = []

    for case in DRY_RUN:

        match = pilot[
            (pilot["tool"] == case["tool"])
            &
            (
                pilot["repo_name"]
                == case["repo_name"]
            )
            &
            (
                pilot["pr_number"]
                == case["pr_number"]
            )
        ]

        if len(match) != 1:
            raise RuntimeError(
                "Expected exactly one pilot row for "
                f"{case['repo_name']} "
                f"PR #{case['pr_number']}, "
                f"found {len(match)}"
            )

        row = match.iloc[0].to_dict()
        row["case_type"] = case["case_type"]

        selected_rows.append(row)

    dry_run_df = pd.DataFrame(
        selected_rows
    )

    dry_run_df.to_csv(
        output_dir /
        "04_dry_run_episodes.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # DuckDB views over local SpecMine data.
    # ---------------------------------------------------------

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

    con.execute(
        f"""
        CREATE VIEW spec_history AS
        SELECT *
        FROM read_parquet(
            '{sql_path(history_path)}'
        )
        """
    )

    session = github_session()

    episode_results = []
    history_results = []
    changed_file_results = []

    raw_github = []

    fetched_at = datetime.now(
        timezone.utc
    ).isoformat()

    # ---------------------------------------------------------
    # Reconstruct each of the five episodes.
    # ---------------------------------------------------------

    for case in DRY_RUN:

        tool = case["tool"]
        repo_name = case["repo_name"]
        pr_number = case["pr_number"]

        print()
        print("=" * 70)
        print(
            f"{repo_name} PR #{pr_number}"
        )
        print("=" * 70)

        api_base = (
            f"https://api.github.com/repos/"
            f"{repo_name}/pulls/{pr_number}"
        )

        try:

            pr_data = github_get(
                session,
                api_base,
            )

            commits = github_get_all_pages(
                session,
                api_base + "/commits",
            )

            api_status = "ok"
            api_error = None

        except Exception as exc:

            pr_data = {}
            commits = []

            api_status = "error"
            api_error = str(exc)

            print(
                f"GitHub API error: "
                f"{api_error}"
            )

        pr_commit_shas = {
            commit.get("sha")
            for commit in commits
            if commit.get("sha")
        }

        # -----------------------------------------------------
        # Local changed files for this PR.
        # -----------------------------------------------------

        changed_files = con.execute(
            """
            SELECT
                file_path,
                change_status,
                additions,
                deletions,
                is_spec,
                is_code,
                ext

            FROM pr_files

            WHERE tool = ?
              AND repo_name = ?
              AND pr_number = ?

            ORDER BY
                is_spec DESC,
                is_code DESC,
                file_path
            """,
            [
                tool,
                repo_name,
                pr_number,
            ],
        ).fetchdf()

        spec_paths = (
            changed_files[
                changed_files["is_spec"]
                == 1
            ]["file_path"]
            .dropna()
            .unique()
            .tolist()
        )

        code_paths = (
            changed_files[
                changed_files["is_code"]
                == 1
            ]["file_path"]
            .dropna()
            .unique()
            .tolist()
        )

        for _, file_row in (
            changed_files.iterrows()
        ):

            changed_file_results.append(
                {
                    "tool": tool,
                    "repo_name": repo_name,
                    "pr_number": pr_number,
                    "file_path":
                        file_row["file_path"],
                    "change_status":
                        file_row[
                            "change_status"
                        ],
                    "is_spec":
                        file_row["is_spec"],
                    "is_code":
                        file_row["is_code"],
                    "ext":
                        file_row["ext"],
                    "additions":
                        file_row["additions"],
                    "deletions":
                        file_row["deletions"],
                }
            )

        # -----------------------------------------------------
        # Compare each changed spec path to its recorded
        # SpecMine commit history.
        # -----------------------------------------------------

        paths_with_history = 0
        paths_with_pr_commit_match = 0
        paths_with_prior_history = 0

        pr_created_at = pd.to_datetime(
            pr_data.get("created_at"),
            utc=True,
            errors="coerce",
        )

        for spec_path in spec_paths:

            history = con.execute(
                """
                SELECT
                    commit_sha,
                    seq_from_oldest,
                    authored_at,
                    committed_at,
                    parents

                FROM spec_history

                WHERE repo_name = ?
                  AND file_path = ?

                ORDER BY seq_from_oldest
                """,
                [
                    repo_name,
                    spec_path,
                ],
            ).fetchdf()

            history_count = len(history)

            if history_count > 0:
                paths_with_history += 1

            history_shas = set(
                history[
                    "commit_sha"
                ].dropna()
            )

            matched_shas = sorted(
                history_shas.intersection(
                    pr_commit_shas
                )
            )

            if matched_shas:
                paths_with_pr_commit_match += 1

            history_dates = pd.to_datetime(
                history["authored_at"],
                utc=True,
                errors="coerce",
            )

            if (
                pd.notna(pr_created_at)
                and len(history_dates) > 0
            ):

                prior_count = int(
                    (
                        history_dates
                        < pr_created_at
                    ).sum()
                )

            else:
                prior_count = 0

            if prior_count > 0:
                paths_with_prior_history += 1

            first_history = (
                history_dates.min()
                if history_count > 0
                else pd.NaT
            )

            last_history = (
                history_dates.max()
                if history_count > 0
                else pd.NaT
            )

            history_results.append(
                {
                    "tool": tool,
                    "repo_name": repo_name,
                    "pr_number": pr_number,
                    "spec_path": spec_path,
                    "history_commits":
                        history_count,
                    "history_before_pr_created":
                        prior_count,
                    "pr_commit_matches":
                        len(matched_shas),
                    "matched_pr_commit_shas":
                        ";".join(
                            matched_shas
                        ),
                    "first_history":
                        (
                            first_history.isoformat()
                            if pd.notna(
                                first_history
                            )
                            else None
                        ),
                    "last_history":
                        (
                            last_history.isoformat()
                            if pd.notna(
                                last_history
                            )
                            else None
                        ),
                }
            )

        # -----------------------------------------------------
        # Preliminary reconstruction signal.
        #
        # IMPORTANT:
        # This is NOT a Tier A / B / C classification.
        # -----------------------------------------------------

        if len(spec_paths) == 0:

            reconstruction_signal = (
                "no_local_spec_paths"
            )

        elif paths_with_history == 0:

            reconstruction_signal = (
                "spec_history_missing"
            )

        elif (
            paths_with_pr_commit_match > 0
            and paths_with_prior_history > 0
        ):

            reconstruction_signal = (
                "prior_spec_history_and_"
                "spec_changed_in_pr"
            )

        elif paths_with_pr_commit_match > 0:

            reconstruction_signal = (
                "spec_changed_in_pr_"
                "no_prior_history_found"
            )

        else:

            reconstruction_signal = (
                "spec_history_exists_"
                "but_no_pr_commit_match"
            )

        base_sha = None
        head_sha = None
        merge_commit_sha = None

        if pr_data:

            base_sha = (
                pr_data.get("base", {})
                .get("sha")
            )

            head_sha = (
                pr_data.get("head", {})
                .get("sha")
            )

            merge_commit_sha = (
                pr_data.get(
                    "merge_commit_sha"
                )
            )

        episode_results.append(
            {
                "tool": tool,
                "repo_name": repo_name,
                "pr_number": pr_number,
                "case_type":
                    case["case_type"],

                "api_status":
                    api_status,

                "api_error":
                    api_error,

                "github_created_at":
                    pr_data.get(
                        "created_at"
                    ),

                "github_merged_at":
                    pr_data.get(
                        "merged_at"
                    ),

                "base_sha":
                    base_sha,

                "head_sha":
                    head_sha,

                "merge_commit_sha":
                    merge_commit_sha,

                "github_pr_commits":
                    len(commits),

                "local_changed_files":
                    len(changed_files),

                "local_spec_paths":
                    len(spec_paths),

                "local_code_paths":
                    len(code_paths),

                "spec_paths_with_history":
                    paths_with_history,

                "spec_paths_with_prior_history":
                    paths_with_prior_history,

                "spec_paths_with_pr_commit_match":
                    paths_with_pr_commit_match,

                "reconstruction_signal":
                    reconstruction_signal,

                "fetched_at":
                    fetched_at,
            }
        )

        raw_github.append(
            {
                "tool": tool,
                "repo_name": repo_name,
                "pr_number": pr_number,
                "fetched_at": fetched_at,
                "pull_request": pr_data,
                "pr_commits": commits,
            }
        )

        print(
            f"API status: {api_status}"
        )

        print(
            f"PR commits: {len(commits)}"
        )

        print(
            f"Changed spec paths: "
            f"{len(spec_paths)}"
        )

        print(
            f"Changed code paths: "
            f"{len(code_paths)}"
        )

        print(
            f"Spec paths with history: "
            f"{paths_with_history}"
        )

        print(
            f"Spec paths matching a "
            f"PR commit: "
            f"{paths_with_pr_commit_match}"
        )

        print(
            f"Signal: "
            f"{reconstruction_signal}"
        )

    # ---------------------------------------------------------
    # Save outputs.
    # ---------------------------------------------------------

    episodes_df = pd.DataFrame(
        episode_results
    )

    history_df = pd.DataFrame(
        history_results
    )

    changed_files_df = pd.DataFrame(
        changed_file_results
    )

    episodes_path = (
        output_dir /
        "04_reconstruction_summary.csv"
    )

    history_out_path = (
        output_dir /
        "04_spec_history_check.csv"
    )

    files_out_path = (
        output_dir /
        "04_changed_files.csv"
    )

    raw_path = (
        output_dir /
        "04_github_raw.json"
    )

    report_path = (
        output_dir /
        "04_reconstruction_summary.txt"
    )

    episodes_df.to_csv(
        episodes_path,
        index=False,
    )

    history_df.to_csv(
        history_out_path,
        index=False,
    )

    changed_files_df.to_csv(
        files_out_path,
        index=False,
    )

    raw_path.write_text(
        json.dumps(
            raw_github,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = []

    report.append(
        "SPECMINE HISTORICAL "
        "RECONSTRUCTION DRY RUN"
    )

    report.append("=" * 80)

    report.append("")
    report.append(
        "IMPORTANT:"
    )

    report.append(
        "These are preliminary reconstruction "
        "signals, NOT Tier A/B/C classifications."
    )

    report.append("")
    report.append(
        episodes_df.to_string(
            index=False
        )
    )

    report.append("")
    report.append(
        "SPEC PATH HISTORY"
    )

    report.append("-" * 80)

    report.append(
        history_df.to_string(
            index=False
        )
    )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("Dry run finished.")
    print("=" * 70)

    print()
    print("Created:")
    print(episodes_path)
    print(history_out_path)
    print(files_out_path)
    print(raw_path)
    print(report_path)


if __name__ == "__main__":
    main()