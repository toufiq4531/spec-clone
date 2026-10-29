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
            r'python src\01_pr_candidate_inventory.py '
            r'"D:\path\to\specmine-full"'
        )
        sys.exit(1)

    root = Path(sys.argv[1]).resolve()
    data_dir = root / "data"

    required_files = {
        "pull_requests": data_dir / "pull_requests.parquet",
        "pr_files": data_dir / "pr_files.parquet",
        "spec_files": data_dir / "spec_files.parquet",
        "repo_trees": data_dir / "repo_trees.parquet",
    }

    missing = [
        str(path)
        for path in required_files.values()
        if not path.exists()
    ]

    if missing:
        print("ERROR: Required files are missing:")
        for path in missing:
            print(path)
        sys.exit(1)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    con = duckdb.connect()

    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=4")

    # ---------------------------------------------------------
    # Create convenient views over the Parquet files.
    # ---------------------------------------------------------

    con.execute(
        f"""
        CREATE VIEW prs AS
        SELECT *
        FROM read_parquet(
            '{sql_path(required_files["pull_requests"])}'
        )
        """
    )

    con.execute(
        f"""
        CREATE VIEW prf AS
        SELECT *
        FROM read_parquet(
            '{sql_path(required_files["pr_files"])}'
        )
        """
    )

    con.execute(
        f"""
        CREATE VIEW specs AS
        SELECT *
        FROM read_parquet(
            '{sql_path(required_files["spec_files"])}'
        )
        """
    )

    con.execute(
        f"""
        CREATE VIEW trees AS
        SELECT *
        FROM read_parquet(
            '{sql_path(required_files["repo_trees"])}'
        )
        """
    )

    # ---------------------------------------------------------
    # 1. Overall PR population.
    # ---------------------------------------------------------

    core = con.execute(
        """
        SELECT
            COUNT(*) AS total_prs,

            COUNT(DISTINCT repo_name)
                AS total_repositories,

            COUNT(DISTINCT tool)
                AS total_tools,

            SUM(
                CASE
                    WHEN n_spec_files > 0
                    THEN 1 ELSE 0
                END
            ) AS prs_with_visible_spec,

            SUM(
                CASE
                    WHEN touches_code = 1
                    THEN 1 ELSE 0
                END
            ) AS prs_touching_code,

            SUM(
                CASE
                    WHEN n_spec_files > 0
                     AND touches_code = 1
                    THEN 1 ELSE 0
                END
            ) AS candidate_prs,

            SUM(
                CASE
                    WHEN n_spec_files > 0
                     AND touches_code = 1
                     AND is_merged = 1
                    THEN 1 ELSE 0
                END
            ) AS merged_candidate_prs

        FROM prs
        """
    ).fetchdf()

    # ---------------------------------------------------------
    # 2. Candidate PR counts by tool.
    # ---------------------------------------------------------

    tool_summary = con.execute(
        """
        SELECT
            tool,

            COUNT(*) AS total_prs,

            COUNT(DISTINCT repo_name)
                AS total_repositories,

            SUM(
                CASE
                    WHEN n_spec_files > 0
                    THEN 1 ELSE 0
                END
            ) AS spec_prs,

            SUM(
                CASE
                    WHEN touches_code = 1
                    THEN 1 ELSE 0
                END
            ) AS code_prs,

            SUM(
                CASE
                    WHEN n_spec_files > 0
                     AND touches_code = 1
                    THEN 1 ELSE 0
                END
            ) AS candidate_prs,

            SUM(
                CASE
                    WHEN n_spec_files > 0
                     AND touches_code = 1
                     AND is_merged = 1
                    THEN 1 ELSE 0
                END
            ) AS merged_candidate_prs,

            COUNT(
                DISTINCT CASE
                    WHEN n_spec_files > 0
                     AND touches_code = 1
                    THEN repo_name
                END
            ) AS candidate_repositories

        FROM prs

        GROUP BY tool

        ORDER BY candidate_prs DESC
        """
    ).fetchdf()

    # ---------------------------------------------------------
    # 3. Check PR counters against actual pr_files rows.
    # ---------------------------------------------------------

    file_consistency = con.execute(
        """
        WITH file_counts AS (

            SELECT
                tool,
                repo_name,
                pr_number,

                SUM(
                    CASE WHEN is_spec = 1
                    THEN 1 ELSE 0 END
                ) AS actual_spec_files,

                SUM(
                    CASE WHEN is_code = 1
                    THEN 1 ELSE 0 END
                ) AS actual_code_files

            FROM prf

            GROUP BY
                tool,
                repo_name,
                pr_number
        ),

        candidates AS (

            SELECT *
            FROM prs

            WHERE n_spec_files > 0
              AND touches_code = 1
        )

        SELECT
            COUNT(*) AS candidate_prs,

            SUM(
                CASE
                    WHEN COALESCE(
                        actual_spec_files, 0
                    ) > 0
                    AND COALESCE(
                        actual_code_files, 0
                    ) > 0
                    THEN 1 ELSE 0
                END
            ) AS confirmed_from_pr_files,

            SUM(
                CASE
                    WHEN COALESCE(
                        actual_spec_files, 0
                    ) = 0
                    OR COALESCE(
                        actual_code_files, 0
                    ) = 0
                    THEN 1 ELSE 0
                END
            ) AS inconsistent_candidates

        FROM candidates c

        LEFT JOIN file_counts f
          ON c.tool = f.tool
         AND c.repo_name = f.repo_name
         AND c.pr_number = f.pr_number
        """
    ).fetchdf()

    # ---------------------------------------------------------
    # 4. files_status distribution among candidates.
    # ---------------------------------------------------------

    files_status = con.execute(
        """
        SELECT
            COALESCE(files_status, 'NULL')
                AS files_status,

            COUNT(*) AS candidate_prs

        FROM prs

        WHERE n_spec_files > 0
          AND touches_code = 1

        GROUP BY files_status

        ORDER BY candidate_prs DESC
        """
    ).fetchdf()

    # ---------------------------------------------------------
    # 5. Repository language.
    #
    # spec_files repeats repository metadata, so reduce it
    # to one row per repository first.
    # ---------------------------------------------------------

    language_summary = con.execute(
        """
        WITH repo_language AS (

            SELECT
                repo_name,
                MAX(repo_language) AS repo_language

            FROM specs

            GROUP BY repo_name
        ),

        candidates AS (

            SELECT *
            FROM prs

            WHERE n_spec_files > 0
              AND touches_code = 1
        )

        SELECT
            COALESCE(
                r.repo_language,
                'UNKNOWN'
            ) AS repo_language,

            COUNT(*) AS candidate_prs,

            COUNT(
                DISTINCT c.repo_name
            ) AS repositories,

            SUM(
                CASE
                    WHEN c.is_merged = 1
                    THEN 1 ELSE 0
                END
            ) AS merged_candidate_prs

        FROM candidates c

        LEFT JOIN repo_language r
          ON c.repo_name = r.repo_name

        GROUP BY
            COALESCE(
                r.repo_language,
                'UNKNOWN'
            )

        ORDER BY candidate_prs DESC
        """
    ).fetchdf()

    # ---------------------------------------------------------
    # 6. Can candidate PR spec paths be found in spec_files?
    # ---------------------------------------------------------

    spec_path_coverage = con.execute(
        """
        WITH candidate_keys AS (

            SELECT
                tool,
                repo_name,
                pr_number

            FROM prs

            WHERE n_spec_files > 0
              AND touches_code = 1
        ),

        candidate_spec_paths AS (

            SELECT DISTINCT
                f.tool,
                f.repo_name,
                f.pr_number,
                f.file_path

            FROM prf f

            INNER JOIN candidate_keys c
              ON f.tool = c.tool
             AND f.repo_name = c.repo_name
             AND f.pr_number = c.pr_number

            WHERE f.is_spec = 1
        ),

        spec_meta AS (

            SELECT
                repo_name,
                file_path,

                MAX(
                    CASE
                        WHEN file_url_sha16
                             IS NOT NULL
                        THEN 1 ELSE 0
                    END
                ) AS exists_in_spec_files,

                MAX(
                    CASE
                        WHEN COALESCE(
                            file_total_commits,
                            0
                        ) > 0
                        THEN 1 ELSE 0
                    END
                ) AS has_commit_history

            FROM specs

            GROUP BY
                repo_name,
                file_path
        )

        SELECT
            COUNT(*) AS candidate_spec_paths,

            SUM(
                CASE
                    WHEN s.exists_in_spec_files = 1
                    THEN 1 ELSE 0
                END
            ) AS paths_found_in_spec_files,

            SUM(
                CASE
                    WHEN s.has_commit_history = 1
                    THEN 1 ELSE 0
                END
            ) AS paths_with_commit_history

        FROM candidate_spec_paths c

        LEFT JOIN spec_meta s
          ON c.repo_name = s.repo_name
         AND c.file_path = s.file_path
        """
    ).fetchdf()

    # ---------------------------------------------------------
    # 7. Repository tree metadata coverage.
    # ---------------------------------------------------------

    tree_coverage = con.execute(
        """
        WITH candidate_repos AS (

            SELECT DISTINCT repo_name

            FROM prs

            WHERE n_spec_files > 0
              AND touches_code = 1
        ),

        tree_meta AS (

            SELECT
                repo_name,

                MAX(status) AS tree_status,

                MAX(
                    COALESCE(
                        truncated,
                        0
                    )
                ) AS truncated

            FROM trees

            GROUP BY repo_name
        )

        SELECT
            COUNT(*) AS candidate_repositories,

            SUM(
                CASE
                    WHEN t.repo_name IS NOT NULL
                    THEN 1 ELSE 0
                END
            ) AS repositories_with_tree_metadata,

            SUM(
                CASE
                    WHEN COALESCE(
                        t.truncated,
                        0
                    ) = 1
                    THEN 1 ELSE 0
                END
            ) AS repositories_with_truncated_tree

        FROM candidate_repos c

        LEFT JOIN tree_meta t
          ON c.repo_name = t.repo_name
        """
    ).fetchdf()

    # ---------------------------------------------------------
    # 8. Build candidate-level file.
    # ---------------------------------------------------------

    candidates = con.execute(
        """
        WITH file_counts AS (

            SELECT
                tool,
                repo_name,
                pr_number,

                SUM(
                    CASE WHEN is_spec = 1
                    THEN 1 ELSE 0 END
                ) AS actual_spec_files,

                SUM(
                    CASE WHEN is_code = 1
                    THEN 1 ELSE 0 END
                ) AS actual_code_files

            FROM prf

            GROUP BY
                tool,
                repo_name,
                pr_number
        ),

        repo_language AS (

            SELECT
                repo_name,
                MAX(repo_language)
                    AS repo_language

            FROM specs

            GROUP BY repo_name
        ),

        tree_meta AS (

            SELECT
                repo_name,
                MAX(status) AS tree_status,
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
            p.state,
            p.is_merged,
            p.created_at,
            p.merged_at,

            p.n_files,
            p.n_spec_files,
            p.n_code_files,
            p.touches_code,

            COALESCE(
                f.actual_spec_files,
                0
            ) AS actual_spec_files,

            COALESCE(
                f.actual_code_files,
                0
            ) AS actual_code_files,

            p.additions,
            p.deletions,
            p.files_status,

            r.repo_language,

            t.tree_status,
            t.tree_truncated

        FROM prs p

        LEFT JOIN file_counts f
          ON p.tool = f.tool
         AND p.repo_name = f.repo_name
         AND p.pr_number = f.pr_number

        LEFT JOIN repo_language r
          ON p.repo_name = r.repo_name

        LEFT JOIN tree_meta t
          ON p.repo_name = t.repo_name

        WHERE p.n_spec_files > 0
          AND p.touches_code = 1

        ORDER BY
            p.tool,
            p.repo_name,
            p.pr_number
        """
    ).fetchdf()

    # ---------------------------------------------------------
    # Save outputs.
    # ---------------------------------------------------------

    tool_summary.to_csv(
        output_dir / "01_tool_summary.csv",
        index=False
    )

    language_summary.to_csv(
        output_dir / "01_language_summary.csv",
        index=False
    )

    candidates.to_csv(
        output_dir / "01_candidate_prs.csv",
        index=False
    )

    report = []

    report.append(
        "SPECMINE PR CANDIDATE INVENTORY"
    )
    report.append("=" * 80)

    report.append("")
    report.append(
        "Candidate definition:"
    )
    report.append(
        "n_spec_files > 0 AND touches_code = 1"
    )

    report.append("")
    report.append("CORE COUNTS")
    report.append("-" * 80)
    report.append(
        core.to_string(index=False)
    )

    report.append("")
    report.append("BY TOOL")
    report.append("-" * 80)
    report.append(
        tool_summary.to_string(index=False)
    )

    report.append("")
    report.append(
        "PR_FILES CONSISTENCY CHECK"
    )
    report.append("-" * 80)
    report.append(
        file_consistency.to_string(index=False)
    )

    report.append("")
    report.append(
        "FILES_STATUS AMONG CANDIDATES"
    )
    report.append("-" * 80)
    report.append(
        files_status.to_string(index=False)
    )

    report.append("")
    report.append(
        "CANDIDATE LANGUAGE DISTRIBUTION"
    )
    report.append("-" * 80)
    report.append(
        language_summary.to_string(index=False)
    )

    report.append("")
    report.append(
        "SPEC PATH COVERAGE"
    )
    report.append("-" * 80)
    report.append(
        spec_path_coverage.to_string(
            index=False
        )
    )

    report.append("")
    report.append(
        "REPOSITORY TREE COVERAGE"
    )
    report.append("-" * 80)
    report.append(
        tree_coverage.to_string(index=False)
    )

    report_path = (
        output_dir /
        "01_pr_candidate_inventory.txt"
    )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print()
    print("PR candidate inventory complete.")
    print()

    print(core.to_string(index=False))

    print()
    print("Candidate PRs by tool:")
    print(tool_summary.to_string(index=False))

    print()
    print("Candidate PRs by language:")
    print(
        language_summary.head(15).to_string(
            index=False
        )
    )

    print()
    print(
        f"Full report: {report_path.resolve()}"
    )

    print(
        "Candidate file:",
        (
            output_dir /
            "01_candidate_prs.csv"
        ).resolve()
    )


if __name__ == "__main__":
    main()