from pathlib import Path
from urllib.parse import quote
import json
import os
import sys

import pandas as pd
import requests


API_VERSION = "2026-03-10"

DRY_RUN_FILE = Path(
    "outputs/04_reconstruction_summary.csv"
)

CHANGED_FILES_FILE = Path(
    "outputs/04_changed_files.csv"
)

RAW_GITHUB_FILE = Path(
    "outputs/04_github_raw.json"
)

OUTPUT_DIR = Path("outputs")


def github_session():

    session = requests.Session()

    session.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent":
                "specmine-clones-before-code",
        }
    )

    token = os.environ.get("GITHUB_TOKEN")

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

    if allow_404 and response.status_code == 404:
        return None

    if response.status_code != 200:

        raise RuntimeError(
            f"GitHub request failed\n"
            f"Status: {response.status_code}\n"
            f"URL: {response.url}\n"
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
                f"Expected list from {url}"
            )

        items.extend(data)

        if len(data) < 100:
            break

        page += 1

    return items


def file_names(file_data):
    """
    Return current and previous filename.

    previous_filename matters for renamed files.
    """

    names = set()

    filename = file_data.get("filename")

    if filename:
        names.add(filename)

    previous = file_data.get(
        "previous_filename"
    )

    if previous:
        names.add(previous)

    return names


def exists_at_revision(
    session,
    repo_name,
    path,
    ref,
):

    encoded_path = quote(
        path,
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
        params={"ref": ref},
        allow_404=True,
    )

    return data is not None


def main():

    required = [
        DRY_RUN_FILE,
        CHANGED_FILES_FILE,
        RAW_GITHUB_FILE,
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

    dry_run = pd.read_csv(
        DRY_RUN_FILE
    )

    changed = pd.read_csv(
        CHANGED_FILES_FILE
    )

    raw = json.loads(
        RAW_GITHUB_FILE.read_text(
            encoding="utf-8"
        )
    )

    session = github_session()

    # -----------------------------------------------------
    # Index raw GitHub responses from Step 4.
    # -----------------------------------------------------

    raw_index = {}

    for item in raw:

        key = (
            item["repo_name"],
            int(item["pr_number"]),
        )

        raw_index[key] = item

    summary_rows = []
    timeline_rows = []
    base_rows = []

    # -----------------------------------------------------
    # Five dry-run episodes
    # -----------------------------------------------------

    for _, episode in dry_run.iterrows():

        repo_name = episode["repo_name"]
        pr_number = int(
            episode["pr_number"]
        )

        base_sha = episode["base_sha"]

        key = (
            repo_name,
            pr_number,
        )

        print()
        print("=" * 75)
        print(
            f"{repo_name} PR #{pr_number}"
        )
        print("=" * 75)

        local_files = changed[
            (changed["repo_name"] == repo_name)
            &
            (
                changed["pr_number"]
                == pr_number
            )
        ].copy()

        spec_paths = set(
            local_files.loc[
                local_files["is_spec"] == 1,
                "file_path",
            ].dropna()
        )

        code_paths = set(
            local_files.loc[
                local_files["is_code"] == 1,
                "file_path",
            ].dropna()
        )

        raw_item = raw_index.get(key)

        if raw_item is None:
            raise RuntimeError(
                f"No Step-4 GitHub data for "
                f"{repo_name} #{pr_number}"
            )

        pr_commits = raw_item.get(
            "pr_commits",
            [],
        )

        # -------------------------------------------------
        # Get GitHub's PR-file representation too.
        #
        # This gives us previous_filename for renames.
        # -------------------------------------------------

        pr_files_url = (
            f"https://api.github.com/repos/"
            f"{repo_name}/pulls/"
            f"{pr_number}/files"
        )

        try:

            github_pr_files = (
                github_get_all_pages(
                    session,
                    pr_files_url,
                )
            )

            pr_files_status = "ok"

        except Exception as exc:

            github_pr_files = []

            pr_files_status = (
                "github_diff_unavailable"
            )

            print(
                "WARNING: GitHub could not "
                "generate the PR file diff."
            )

            print(str(exc))

        # -------------------------------------------------
        # Did each changed spec path already exist at
        # the PR base revision?
        # -------------------------------------------------

        spec_exists_results = []

        for spec_path in sorted(
            spec_paths
        ):

            matching_file = None

            for gh_file in github_pr_files:

                names = file_names(
                    gh_file
                )

                if spec_path in names:
                    matching_file = gh_file
                    break

            lookup_path = spec_path
            gh_status = None
            previous_filename = None

            if matching_file is not None:

                gh_status = matching_file.get(
                    "status"
                )

                previous_filename = (
                    matching_file.get(
                        "previous_filename"
                    )
                )

                # If GitHub says the file was renamed,
                # check the OLD pathname at the base.
                if (
                    gh_status == "renamed"
                    and previous_filename
                ):
                    lookup_path = (
                        previous_filename
                    )

            exists = exists_at_revision(
                session,
                repo_name,
                lookup_path,
                base_sha,
            )

            spec_exists_results.append(
                exists
            )

            base_rows.append(
                {
                    "repo_name":
                        repo_name,

                    "pr_number":
                        pr_number,

                    "spec_path":
                        spec_path,

                    "github_status":
                        gh_status,

                    "previous_filename":
                        previous_filename,

                    "base_lookup_path":
                        lookup_path,

                    "exists_at_base":
                        int(exists),
                }
            )

        # -------------------------------------------------
        # Walk through the actual PR commits in order.
        # -------------------------------------------------

        first_spec_position = None
        first_code_position = None

        for position, commit in enumerate(
            pr_commits,
            start=1,
        ):

            sha = commit.get("sha")

            commit_url = (
                f"https://api.github.com/repos/"
                f"{repo_name}/commits/{sha}"
            )

            detail = github_get(
                session,
                commit_url,
            )

            commit_files = detail.get(
                "files",
                [],
            )

            touched_names = set()

            for file_data in commit_files:
                touched_names.update(
                    file_names(file_data)
                )

            touched_specs = sorted(
                spec_paths.intersection(
                    touched_names
                )
            )

            touched_code = sorted(
                code_paths.intersection(
                    touched_names
                )
            )

            touches_spec = (
                len(touched_specs) > 0
            )

            touches_code = (
                len(touched_code) > 0
            )

            if (
                touches_spec
                and first_spec_position
                is None
            ):
                first_spec_position = (
                    position
                )

            if (
                touches_code
                and first_code_position
                is None
            ):
                first_code_position = (
                    position
                )

            commit_info = detail.get(
                "commit",
                {},
            )

            commit_author = (
                commit_info.get(
                    "author",
                    {}
                )
            )

            timeline_rows.append(
                {
                    "repo_name":
                        repo_name,

                    "pr_number":
                        pr_number,

                    "position":
                        position,

                    "commit_sha":
                        sha,

                    "commit_date":
                        commit_author.get(
                            "date"
                        ),

                    "commit_message":
                        commit_info.get(
                            "message"
                        ),

                    "n_changed_files":
                        len(commit_files),

                    "touches_spec":
                        int(
                            touches_spec
                        ),

                    "touches_code":
                        int(
                            touches_code
                        ),

                    "n_spec_paths_touched":
                        len(
                            touched_specs
                        ),

                    "n_code_paths_touched":
                        len(
                            touched_code
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

        # -------------------------------------------------
        # Preliminary ordering signal.
        #
        # Still NOT Tier A / B / C.
        # -------------------------------------------------

        if len(pr_commits) == 0:

            ordering_signal = (
                "pr_commit_history_unavailable"
            )

        elif (
            first_spec_position is None
            and first_code_position is None
        ):

            ordering_signal = (
                "spec_and_code_touch_not_found"
            )

        elif first_spec_position is None:

            ordering_signal = (
                "spec_touch_not_found"
            )

        elif first_code_position is None:

            ordering_signal = (
                "code_touch_not_found"
            )

        elif (
            first_spec_position
            < first_code_position
        ):

            ordering_signal = (
                "spec_commit_before_code_commit"
            )

        elif (
            first_code_position
            < first_spec_position
        ):

            ordering_signal = (
                "code_commit_before_spec_commit"
            )

        else:

            ordering_signal = (
                "spec_and_code_first_touched_"
                "same_commit"
            )

        base_count = sum(
            spec_exists_results
        )

        total_specs = len(
            spec_exists_results
        )

        if total_specs == 0:

            base_signal = (
                "no_spec_paths"
            )

        elif base_count == 0:

            base_signal = (
                "no_changed_spec_path_"
                "exists_at_base"
            )

        elif base_count == total_specs:

            base_signal = (
                "all_changed_spec_paths_"
                "exist_at_base"
            )

        else:

            base_signal = (
                "some_changed_spec_paths_"
                "exist_at_base"
            )

        summary_rows.append(
            {
                "tool":
                    episode["tool"],

                "repo_name":
                    repo_name,

                "pr_number":
                    pr_number,

                "case_type":
                    episode[
                        "case_type"
                    ],

                "github_pr_commits":
                    len(
                        pr_commits
                    ),

                "pr_files_status":
                    pr_files_status,

                "local_spec_paths":
                    len(
                        spec_paths
                    ),

                "local_code_paths":
                    len(
                        code_paths
                    ),

                "spec_paths_existing_at_base":
                    base_count,

                "first_spec_commit_position":
                    first_spec_position,

                "first_code_commit_position":
                    first_code_position,

                "base_signal":
                    base_signal,

                "ordering_signal":
                    ordering_signal,
            }
        )

        print(
            f"PR commits: "
            f"{len(pr_commits)}"
        )

        print(
            f"Spec paths existing at base: "
            f"{base_count}/{total_specs}"
        )

        print(
            f"First spec commit: "
            f"{first_spec_position}"
        )

        print(
            f"First code commit: "
            f"{first_code_position}"
        )

        print(
            f"Ordering: "
            f"{ordering_signal}"
        )

    # -----------------------------------------------------
    # Outputs
    # -----------------------------------------------------

    summary = pd.DataFrame(
        summary_rows
    )

    timeline = pd.DataFrame(
        timeline_rows
    )

    base_presence = pd.DataFrame(
        base_rows
    )

    summary_path = (
        OUTPUT_DIR /
        "05_ordering_summary.csv"
    )

    timeline_path = (
        OUTPUT_DIR /
        "05_commit_timeline.csv"
    )

    base_path = (
        OUTPUT_DIR /
        "05_spec_base_presence.csv"
    )

    text_path = (
        OUTPUT_DIR /
        "05_ordering_summary.txt"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    timeline.to_csv(
        timeline_path,
        index=False,
    )

    base_presence.to_csv(
        base_path,
        index=False,
    )

    report = []

    report.append(
        "SPECMINE COMMIT-ORDER DRY RUN"
    )

    report.append("=" * 80)

    report.append("")
    report.append(
        "IMPORTANT: these are temporal "
        "signals, NOT Tier A/B/C labels."
    )

    report.append("")
    report.append(
        summary.to_string(
            index=False
        )
    )

    report.append("")
    report.append(
        "COMMIT TIMELINE"
    )

    report.append("-" * 80)

    report.append(
        timeline.to_string(
            index=False
        )
    )

    report.append("")
    report.append(
        "SPEC PRESENCE AT PR BASE"
    )

    report.append("-" * 80)

    report.append(
        base_presence.to_string(
            index=False
        )
    )

    text_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print()
    print("=" * 75)
    print("Step 5 complete.")
    print("=" * 75)

    print(summary.to_string(index=False))

    print()
    print("Created:")
    print(summary_path)
    print(timeline_path)
    print(base_path)
    print(text_path)


if __name__ == "__main__":
    main()