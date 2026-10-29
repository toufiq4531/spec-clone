from pathlib import Path
import sys

import duckdb
import pandas as pd


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def main():

    if len(sys.argv) != 2:
        print("Usage:")
        print(
            r'python src\02_build_pilot_frame.py '
            r'"D:\path\to\specmine-full"'
        )
        sys.exit(1)

    root = Path(sys.argv[1]).resolve()
    data = root / "data"

    files = {
        "prs": data / "pull_requests.parquet",
        "pr_files": data / "pr_files.parquet",
        "specs": data / "spec_files.parquet",
        "history": data / "spec_file_commits.parquet",
        "trees": data / "repo_trees.parquet",
    }

    missing = [
        str(path)
        for path in files.values()
        if not path.exists()
    ]

    if missing:
        print("ERROR: Missing required files:")
        for path in missing:
            print(path)
        sys.exit(1)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    con = duckdb.connect()

    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=4")

    # ---------------------------------------------------------
    # Views
    # ---------------------------------------------------------

    for name, path in files.items():
        con.execute(
            f"""
            CREATE VIEW {name} AS
            SELECT *
            FROM read_parquet('{sql_path(path)}')
            """
        )

    # ---------------------------------------------------------
    # Build PR-level pilot frame
    # ---------------------------------------------------------

    frame = con.execute(
        """
        WITH

        candidates AS (
            SELECT *
            FROM prs
            WHERE n_spec_files > 0
              AND touches_code = 1
        ),

        candidate_counts AS (
            SELECT
                repo_name,
                COUNT(*) AS candidate_prs_in_repo
            FROM candidates
            GROUP BY repo_name
        ),

        changed_spec_paths AS (
            SELECT DISTINCT
                f.tool,
                f.repo_name,
                f.pr_number,
                f.file_path
            FROM pr_files f

            INNER JOIN candidates c
              ON f.tool = c.tool
             AND f.repo_name = c.repo_name
             AND f.pr_number = c.pr_number

            WHERE f.is_spec = 1
        ),

        spec_census AS (
            SELECT
                repo_name,
                file_path,

                MAX(
                    CASE
                        WHEN file_url_sha16 IS NOT NULL
                        THEN 1 ELSE 0
                    END
                ) AS found_in_spec_census

            FROM specs

            GROUP BY
                repo_name,
                file_path
        ),

        history_by_path AS (
            SELECT
                repo_name,
                file_path,

                COUNT(
                    DISTINCT commit_sha
                ) AS history_commits,

                MIN(authored_at)
                    AS first_history_date,

                MAX(authored_at)
                    AS last_history_date

            FROM history

            GROUP BY
                repo_name,
                file_path
        ),

        pr_spec_history AS (
            SELECT
                s.tool,
                s.repo_name,
                s.pr_number,

                COUNT(*)
                    AS changed_spec_paths,

                SUM(
                    CASE
                        WHEN c.found_in_spec_census = 1
                        THEN 1 ELSE 0
                    END
                ) AS paths_found_in_spec_census,

                SUM(
                    CASE
                        WHEN COALESCE(
                            h.history_commits,
                            0
                        ) > 0
                        THEN 1 ELSE 0
                    END
                ) AS paths_with_history,

                SUM(
                    COALESCE(
                        h.history_commits,
                        0
                    )
                ) AS total_spec_history_commits,

                MIN(h.first_history_date)
                    AS earliest_spec_history,

                MAX(h.last_history_date)
                    AS latest_spec_history

            FROM changed_spec_paths s

            LEFT JOIN spec_census c
              ON s.repo_name = c.repo_name
             AND s.file_path = c.file_path

            LEFT JOIN history_by_path h
              ON s.repo_name = h.repo_name
             AND s.file_path = h.file_path

            GROUP BY
                s.tool,
                s.repo_name,
                s.pr_number
        ),

        code_counts AS (
            SELECT
                tool,
                repo_name,
                pr_number,

                COUNT(*)
                    AS changed_code_files

            FROM pr_files

            WHERE is_code = 1

            GROUP BY
                tool,
                repo_name,
                pr_number
        ),

        code_extension_counts AS (
            SELECT
                tool,
                repo_name,
                pr_number,

                COALESCE(
                    NULLIF(ext, ''),
                    '[none]'
                ) AS ext,

                COUNT(*) AS n

            FROM pr_files

            WHERE is_code = 1

            GROUP BY
                tool,
                repo_name,
                pr_number,
                COALESCE(
                    NULLIF(ext, ''),
                    '[none]'
                )
        ),

        ranked_extensions AS (
            SELECT
                *,

                ROW_NUMBER() OVER (
                    PARTITION BY
                        tool,
                        repo_name,
                        pr_number

                    ORDER BY
                        n DESC,
                        ext
                ) AS rn

            FROM code_extension_counts
        ),

        dominant_extension AS (
            SELECT
                tool,
                repo_name,
                pr_number,
                ext AS dominant_code_extension,
                n AS dominant_extension_files

            FROM ranked_extensions

            WHERE rn = 1
        ),

        repo_metadata AS (
            SELECT
                repo_name,

                MAX(repo_language)
                    AS repo_language,

                MAX(repo_stars)
                    AS repo_stars,

                MAX(repo_forks)
                    AS repo_forks,

                MAX(repo_size_kb)
                    AS repo_size_kb,

                MAX(repo_fork)
                    AS repo_fork,

                MAX(repo_archived)
                    AS repo_archived,

                MAX(repo_default_branch)
                    AS repo_default_branch

            FROM specs

            GROUP BY repo_name
        ),

        tree_metadata AS (
            SELECT
                repo_name,

                MAX(status)
                    AS tree_status,

                MAX(
                    COALESCE(
                        truncated,
                        0
                    )
                ) AS tree_truncated

            FROM trees

            GROUP BY repo_name
        )

        SELECT
            p.tool,
            p.repo_name,
            p.pr_number,
            p.title,

            p.created_at,
            p.merged_at,

            p.n_files,
            p.n_spec_files,
            p.n_code_files,

            p.additions,
            p.deletions,

            r.repo_language,
            r.repo_stars,
            r.repo_forks,
            r.repo_size_kb,
            r.repo_fork,
            r.repo_archived,
            r.repo_default_branch,

            cc.candidate_prs_in_repo,

            COALESCE(
                h.changed_spec_paths,
                0
            ) AS changed_spec_paths,

            COALESCE(
                h.paths_found_in_spec_census,
                0
            ) AS paths_found_in_spec_census,

            COALESCE(
                h.paths_with_history,
                0
            ) AS paths_with_history,

            COALESCE(
                h.total_spec_history_commits,
                0
            ) AS total_spec_history_commits,

            h.earliest_spec_history,
            h.latest_spec_history,

            COALESCE(
                code.changed_code_files,
                0
            ) AS changed_code_files,

            ext.dominant_code_extension,
            ext.dominant_extension_files,

            tree.tree_status,
            tree.tree_truncated,

            CASE
                WHEN COALESCE(
                    h.paths_with_history,
                    0
                ) > 0
                THEN 1 ELSE 0
            END AS has_any_spec_history,

            CASE
                WHEN h.changed_spec_paths > 0
                 AND h.paths_with_history
                     = h.changed_spec_paths
                THEN 1 ELSE 0
            END AS has_all_spec_history,

            CASE
                WHEN COALESCE(
                    h.paths_with_history,
                    0
                ) > 0
                 AND COALESCE(
                    tree.tree_truncated,
                    0
                ) = 0
                 AND COALESCE(
                    code.changed_code_files,
                    0
                ) > 0
                THEN 1 ELSE 0
            END AS pilot_ready_basic

        FROM candidates p

        LEFT JOIN pr_spec_history h
          ON p.tool = h.tool
         AND p.repo_name = h.repo_name
         AND p.pr_number = h.pr_number

        LEFT JOIN code_counts code
          ON p.tool = code.tool
         AND p.repo_name = code.repo_name
         AND p.pr_number = code.pr_number

        LEFT JOIN dominant_extension ext
          ON p.tool = ext.tool
         AND p.repo_name = ext.repo_name
         AND p.pr_number = ext.pr_number

        LEFT JOIN repo_metadata r
          ON p.repo_name = r.repo_name

        LEFT JOIN candidate_counts cc
          ON p.repo_name = cc.repo_name

        LEFT JOIN tree_metadata tree
          ON p.repo_name = tree.repo_name

        ORDER BY
            p.tool,
            p.repo_name,
            p.pr_number
        """
    ).fetchdf()

    # ---------------------------------------------------------
    # Save complete frame
    # ---------------------------------------------------------

    frame_path = output_dir / "02_pilot_frame.csv"

    frame.to_csv(
        frame_path,
        index=False
    )

    # ---------------------------------------------------------
    # Summaries
    # ---------------------------------------------------------

    total = len(frame)

    any_history = int(
        frame["has_any_spec_history"].sum()
    )

    all_history = int(
        frame["has_all_spec_history"].sum()
    )

    pilot_ready = int(
        frame["pilot_ready_basic"].sum()
    )

    total_repos = frame["repo_name"].nunique()

    ready_repos = frame.loc[
        frame["pilot_ready_basic"] == 1,
        "repo_name"
    ].nunique()

    tool_summary = (
        frame.groupby("tool")
        .agg(
            candidate_prs=("pr_number", "size"),
            repositories=("repo_name", "nunique"),
            any_history=(
                "has_any_spec_history",
                "sum"
            ),
            all_history=(
                "has_all_spec_history",
                "sum"
            ),
            pilot_ready=(
                "pilot_ready_basic",
                "sum"
            ),
        )
        .sort_values(
            "candidate_prs",
            ascending=False
        )
        .reset_index()
    )

    language_summary = (
        frame.groupby(
            "repo_language",
            dropna=False
        )
        .agg(
            candidate_prs=("pr_number", "size"),
            repositories=("repo_name", "nunique"),
            pilot_ready=(
                "pilot_ready_basic",
                "sum"
            ),
        )
        .sort_values(
            "candidate_prs",
            ascending=False
        )
        .reset_index()
    )

    repo_summary = (
        frame.groupby(
            [
                "repo_name",
                "tool",
                "repo_language",
            ],
            dropna=False
        )
        .agg(
            candidate_prs=("pr_number", "size"),
            pilot_ready=(
                "pilot_ready_basic",
                "sum"
            ),
            repo_stars=("repo_stars", "max"),
            repo_size_kb=("repo_size_kb", "max"),
        )
        .sort_values(
            "candidate_prs",
            ascending=False
        )
        .reset_index()
    )

    tool_summary.to_csv(
        output_dir /
        "02_pilot_tool_summary.csv",
        index=False
    )

    language_summary.to_csv(
        output_dir /
        "02_pilot_language_summary.csv",
        index=False
    )

    repo_summary.to_csv(
        output_dir /
        "02_pilot_repo_summary.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Text report
    # ---------------------------------------------------------

    report = []

    report.append(
        "SPECMINE PILOT SAMPLING FRAME"
    )
    report.append("=" * 80)

    report.append("")
    report.append("OVERALL")
    report.append("-" * 80)

    report.append(
        f"Candidate PRs: {total:,}"
    )

    report.append(
        f"Candidate repositories: "
        f"{total_repos:,}"
    )

    report.append(
        f"PRs with ANY recoverable "
        f"spec history: "
        f"{any_history:,}"
    )

    report.append(
        f"PRs with ALL changed spec "
        f"paths having history: "
        f"{all_history:,}"
    )

    report.append(
        f"Basic pilot-ready PRs: "
        f"{pilot_ready:,}"
    )

    report.append(
        f"Repositories with at least "
        f"one basic pilot-ready PR: "
        f"{ready_repos:,}"
    )

    report.append("")
    report.append("BY TOOL")
    report.append("-" * 80)

    report.append(
        tool_summary.to_string(index=False)
    )

    report.append("")
    report.append("BY LANGUAGE")
    report.append("-" * 80)

    report.append(
        language_summary.to_string(
            index=False
        )
    )

    report.append("")
    report.append(
        "TOP 30 REPOSITORIES BY "
        "CANDIDATE COUNT"
    )
    report.append("-" * 80)

    report.append(
        repo_summary.head(30).to_string(
            index=False
        )
    )

    report_path = (
        output_dir /
        "02_pilot_frame_summary.txt"
    )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print()
    print("Pilot frame built successfully.")
    print()

    print(
        f"Candidate PRs: {total:,}"
    )

    print(
        f"With any spec history: "
        f"{any_history:,}"
    )

    print(
        f"With all spec history: "
        f"{all_history:,}"
    )

    print(
        f"Basic pilot-ready: "
        f"{pilot_ready:,}"
    )

    print(
        f"Pilot-ready repositories: "
        f"{ready_repos:,}"
    )

    print()
    print(
        f"Frame: {frame_path.resolve()}"
    )

    print(
        f"Summary: {report_path.resolve()}"
    )


if __name__ == "__main__":
    main()