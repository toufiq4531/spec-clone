from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("outputs/02_pilot_frame.csv")
OUTPUT_DIR = Path("outputs")

RANDOM_SEED = 20261008


# ---------------------------------------------------------
# Desired repository composition.
#
# This is a PURPOSEFUL feasibility sample.
# It is NOT a prevalence sample.
# ---------------------------------------------------------

TARGETS = {
    ("openspec", "TypeScript"): 3,
    ("openspec", "Python"): 2,
    ("openspec", "Go"): 2,
    ("openspec", "Rust"): 1,
    ("openspec", "C#"): 1,
    ("openspec", "Elixir"): 1,

    ("speckit", "TypeScript"): 2,
    ("speckit", "Python"): 2,
    ("speckit", "Go"): 2,
    ("speckit", "Rust"): 1,
    ("speckit", "C#"): 1,

    ("claude_code", "Python"): 1,
    ("claude_code", "JavaScript"): 1,

    ("conductor", "TypeScript"): 1,
    ("conductor", "Python"): 1,

    ("moai", "Go"): 1,

    ("codex", "Rust"): 1,

    ("trae", "Python"): 1,
}


# Two repositories are deliberately forced into the pilot.
#
# 1. epam/... contains a PR with no recoverable spec history.
# 2. carverauto/... has truncated tree metadata.
#
# These are stress cases for the feasibility audit.
FORCED_REPOS = {
    ("openspec", "TypeScript"): [
        "epam/ai-dial-admin-frontend"
    ],

    ("openspec", "Elixir"): [
        "carverauto/serviceradar"
    ],
}


# Force the known no-history PR into the pilot.
FORCED_PRS = {
    ("epam/ai-dial-admin-frontend", 3846)
}


def evenly_spaced_rows(df, n):
    """
    Pick rows across the range of change sizes.

    This gives us small, medium and large episodes
    instead of always picking the easiest ones.
    """

    if n <= 0 or len(df) == 0:
        return []

    df = df.sort_values(
        ["change_volume", "pr_number"]
    )

    if len(df) <= n:
        return list(df.index)

    if n == 1:
        positions = [len(df) // 2]

    else:
        positions = (
            np.linspace(
                0,
                len(df) - 1,
                n
            )
            .round()
            .astype(int)
            .tolist()
        )

    selected = []

    for position in positions:
        index = df.index[position]

        if index not in selected:
            selected.append(index)

    # Safety fill.
    for index in df.index:

        if len(selected) >= n:
            break

        if index not in selected:
            selected.append(index)

    return selected[:n]


def build_repo_table(frame):

    repo = (
        frame.groupby(
            [
                "tool",
                "repo_name",
                "repo_language",
            ],
            dropna=False,
        )
        .agg(
            candidate_prs=(
                "pr_number",
                "size"
            ),

            ready_prs=(
                "pilot_ready_basic",
                "sum"
            ),

            complete_history_prs=(
                "has_all_spec_history",
                "sum"
            ),

            repo_stars=(
                "repo_stars",
                "max"
            ),

            repo_size_kb=(
                "repo_size_kb",
                "max"
            ),

            tree_truncated=(
                "tree_truncated",
                "max"
            ),
        )
        .reset_index()
    )

    return repo


def choose_repositories(repo_table):

    selected = []

    for (tool, language), target_n in TARGETS.items():

        candidates = repo_table[
            (repo_table["tool"] == tool)
            &
            (
                repo_table["repo_language"]
                == language
            )
        ].copy()

        forced_names = FORCED_REPOS.get(
            (tool, language),
            []
        )

        # -----------------------------------------
        # Add forced stress-case repositories.
        # -----------------------------------------

        chosen_names = []

        for repo_name in forced_names:

            row = candidates[
                candidates["repo_name"]
                == repo_name
            ]

            if len(row) == 0:
                raise ValueError(
                    f"Forced repository not found: "
                    f"{repo_name}"
                )

            selected.append(
                row.iloc[0].to_dict()
            )

            chosen_names.append(repo_name)

        remaining_n = (
            target_n - len(chosen_names)
        )

        if remaining_n <= 0:
            continue

        candidates = candidates[
            ~candidates["repo_name"].isin(
                chosen_names
            )
        ].copy()

        # Prefer repositories that have at least
        # four candidate episodes because we want
        # roughly four PRs from each repository.
        preferred = candidates[
            (candidates["candidate_prs"] >= 4)
            &
            (candidates["ready_prs"] > 0)
        ].copy()

        if len(preferred) >= remaining_n:
            candidates = preferred

        else:
            usable = candidates[
                candidates["ready_prs"] > 0
            ]

            if len(usable) >= remaining_n:
                candidates = usable

        # -----------------------------------------
        # Sort from low episode count to high.
        #
        # Selecting evenly across this list gives
        # us small and large project situations.
        # -----------------------------------------

        candidates = candidates.sort_values(
            [
                "candidate_prs",
                "repo_size_kb",
                "repo_name",
            ]
        )

        if len(candidates) < remaining_n:

            raise ValueError(
                f"Not enough repositories for "
                f"{tool} / {language}"
            )

        if remaining_n == 1:

            positions = [
                len(candidates) // 2
            ]

        else:

            positions = (
                np.linspace(
                    0,
                    len(candidates) - 1,
                    remaining_n,
                )
                .round()
                .astype(int)
                .tolist()
            )

        for position in positions:

            selected.append(
                candidates.iloc[
                    position
                ].to_dict()
            )

    selected_repos = pd.DataFrame(selected)

    if len(selected_repos) != 25:

        raise ValueError(
            f"Expected 25 repositories, "
            f"got {len(selected_repos)}"
        )

    return selected_repos


def choose_initial_episodes(
    frame,
    selected_repos,
):

    selected_indices = []

    selected_repo_names = set(
        selected_repos["repo_name"]
    )

    pool = frame[
        frame["repo_name"].isin(
            selected_repo_names
        )
    ].copy()

    for repo_name in selected_repos[
        "repo_name"
    ]:

        repo_prs = pool[
            pool["repo_name"] == repo_name
        ].copy()

        target = min(
            4,
            len(repo_prs)
        )

        chosen = []

        # -----------------------------------------
        # Include forced PR if this repository
        # contains one.
        # -----------------------------------------

        for (
            forced_repo,
            forced_pr
        ) in FORCED_PRS:

            if forced_repo != repo_name:
                continue

            match = repo_prs[
                repo_prs["pr_number"]
                == forced_pr
            ]

            if len(match) == 1:

                chosen.extend(
                    match.index.tolist()
                )

        # -----------------------------------------
        # Include one partial-history case when
        # one exists.
        # -----------------------------------------

        if len(chosen) < target:

            partial = repo_prs[
                (repo_prs["history_class"]
                 == "partial")
                &
                (~repo_prs.index.isin(chosen))
            ]

            if len(partial) > 0:

                chosen.extend(
                    evenly_spaced_rows(
                        partial,
                        1
                    )
                )

        # -----------------------------------------
        # Fill remaining positions primarily with
        # complete-history cases.
        # -----------------------------------------

        remaining = target - len(chosen)

        complete = repo_prs[
            (repo_prs["history_class"]
             == "complete")
            &
            (~repo_prs.index.isin(chosen))
        ]

        chosen.extend(
            evenly_spaced_rows(
                complete,
                remaining
            )
        )

        # -----------------------------------------
        # If necessary, fill with anything left.
        # -----------------------------------------

        remaining = target - len(chosen)

        if remaining > 0:

            leftovers = repo_prs[
                ~repo_prs.index.isin(
                    chosen
                )
            ]

            chosen.extend(
                evenly_spaced_rows(
                    leftovers,
                    remaining
                )
            )

        selected_indices.extend(chosen)

    return (
        frame.loc[
            selected_indices
        ].copy()
    )


def fill_to_100(
    frame,
    selected_repos,
    sample,
):

    selected_repo_names = set(
        selected_repos["repo_name"]
    )

    pool = frame[
        frame["repo_name"].isin(
            selected_repo_names
        )
    ].copy()

    used_indices = set(sample.index)

    # Give smaller tool families first chance
    # when we need extra episodes.
    fill_order = [
        "claude_code",
        "conductor",
        "moai",
        "codex",
        "speckit",
        "openspec",
        "trae",
    ]

    while len(sample) < 100:

        added_something = False

        for tool in fill_order:

            if len(sample) >= 100:
                break

            tool_repos = selected_repos[
                selected_repos["tool"]
                == tool
            ]["repo_name"].tolist()

            options = []

            for repo_name in tool_repos:

                current_count = len(
                    sample[
                        sample["repo_name"]
                        == repo_name
                    ]
                )

                # No repository contributes more
                # than five pilot episodes.
                if current_count >= 5:
                    continue

                remaining = pool[
                    (pool["repo_name"]
                     == repo_name)
                    &
                    (~pool.index.isin(
                        used_indices
                    ))
                ]

                if len(remaining) == 0:
                    continue

                options.append(
                    (
                        current_count,
                        -len(remaining),
                        repo_name,
                    )
                )

            if not options:
                continue

            options.sort()

            repo_name = options[0][2]

            remaining = pool[
                (pool["repo_name"]
                 == repo_name)
                &
                (~pool.index.isin(
                    used_indices
                ))
            ]

            complete = remaining[
                remaining["history_class"]
                == "complete"
            ]

            if len(complete) > 0:
                candidates = complete
            else:
                candidates = remaining

            index = evenly_spaced_rows(
                candidates,
                1
            )[0]

            sample = pd.concat(
                [
                    sample,
                    frame.loc[[index]],
                ]
            )

            used_indices.add(index)

            added_something = True

        if not added_something:
            raise RuntimeError(
                "Unable to fill pilot to "
                "100 episodes."
            )

    return sample


def main():

    if not INPUT.exists():

        print(
            "ERROR: Missing:"
        )
        print(INPUT)
        return

    frame = pd.read_csv(INPUT)

    # ---------------------------------------------
    # Useful pilot variables.
    # ---------------------------------------------

    frame["change_volume"] = (
        frame["additions"].fillna(0)
        +
        frame["deletions"].fillna(0)
    )

    frame["history_class"] = np.select(
        [
            frame[
                "has_any_spec_history"
            ] == 0,

            frame[
                "has_all_spec_history"
            ] == 1,
        ],
        [
            "none",
            "complete",
        ],
        default="partial",
    )

    repo_table = build_repo_table(frame)

    selected_repos = choose_repositories(
        repo_table
    )

    sample = choose_initial_episodes(
        frame,
        selected_repos
    )

    sample = fill_to_100(
        frame,
        selected_repos,
        sample
    )

    sample = sample.sort_values(
        [
            "tool",
            "repo_name",
            "pr_number",
        ]
    ).reset_index(drop=True)

    # ---------------------------------------------
    # Save results.
    # ---------------------------------------------

    repo_path = (
        OUTPUT_DIR /
        "03_pilot_repositories.csv"
    )

    sample_path = (
        OUTPUT_DIR /
        "03_pilot_episodes.csv"
    )

    summary_path = (
        OUTPUT_DIR /
        "03_pilot_selection_summary.txt"
    )

    selected_repos.to_csv(
        repo_path,
        index=False
    )

    sample.to_csv(
        sample_path,
        index=False
    )

    report = []

    report.append(
        "SPECMINE PILOT SELECTION"
    )

    report.append("=" * 80)

    report.append("")
    report.append(
        f"Episodes: {len(sample)}"
    )

    report.append(
        f"Repositories: "
        f"{sample['repo_name'].nunique()}"
    )

    report.append("")
    report.append(
        "TOOL DISTRIBUTION"
    )

    report.append("-" * 80)

    report.append(
        sample[
            "tool"
        ].value_counts().to_string()
    )

    report.append("")
    report.append(
        "LANGUAGE DISTRIBUTION"
    )

    report.append("-" * 80)

    report.append(
        sample[
            "repo_language"
        ].value_counts().to_string()
    )

    report.append("")
    report.append(
        "HISTORY CLASS"
    )

    report.append("-" * 80)

    report.append(
        sample[
            "history_class"
        ].value_counts().to_string()
    )

    report.append("")
    report.append(
        "BASIC READINESS"
    )

    report.append("-" * 80)

    report.append(
        sample[
            "pilot_ready_basic"
        ].value_counts().to_string()
    )

    report.append("")
    report.append(
        "SELECTED REPOSITORIES"
    )

    report.append("-" * 80)

    report.append(
        selected_repos[
            [
                "tool",
                "repo_name",
                "repo_language",
                "candidate_prs",
                "ready_prs",
                "complete_history_prs",
                "tree_truncated",
            ]
        ].to_string(index=False)
    )

    summary_path.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print()
    print("Pilot selection complete.")
    print()

    print(
        f"Episodes: {len(sample)}"
    )

    print(
        "Repositories:",
        sample[
            "repo_name"
        ].nunique()
    )

    print()
    print(
        "History classes:"
    )

    print(
        sample[
            "history_class"
        ].value_counts()
    )

    print()
    print(
        "Outputs:"
    )

    print(repo_path)
    print(sample_path)
    print(summary_path)


if __name__ == "__main__":
    main()