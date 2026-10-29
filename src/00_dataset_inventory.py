from pathlib import Path
import sys
import platform
import csv

import duckdb
import pandas
import pyarrow


def safe_sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def human_size(size_bytes: int) -> str:
    value = float(size_bytes)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{value:.2f} TB"


def main():

    if len(sys.argv) != 2:
        print("Usage:")
        print(
            r'python src\00_dataset_inventory.py '
            r'"D:\path\to\specmine-full"'
        )
        sys.exit(1)

    dataset_root = Path(sys.argv[1]).resolve()

    if not dataset_root.exists():
        print("ERROR: Dataset folder does not exist:")
        print(dataset_root)
        sys.exit(1)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    report_path = output_dir / "dataset_inventory.txt"
    csv_path = output_dir / "dataset_inventory.csv"

    parquet_files = sorted(dataset_root.rglob("*.parquet"))

    if not parquet_files:
        print("ERROR: No Parquet files found.")
        sys.exit(1)

    con = duckdb.connect()

    # Keep memory usage conservative on a 16 GB machine.
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=4")

    summary_rows = []
    report = []

    report.append("SPECMINE DATASET INVENTORY")
    report.append("=" * 80)
    report.append("")
    report.append(f"Dataset root: {dataset_root}")
    report.append(f"Python: {platform.python_version()}")
    report.append(f"DuckDB: {duckdb.__version__}")
    report.append(f"Pandas: {pandas.__version__}")
    report.append(f"PyArrow: {pyarrow.__version__}")
    report.append("")
    report.append(f"Parquet files found: {len(parquet_files)}")
    report.append("")

    for parquet_path in parquet_files:

        sql_path = safe_sql_path(parquet_path)
        size_bytes = parquet_path.stat().st_size

        try:

            row_count = con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{sql_path}')
                """
            ).fetchone()[0]

            description = con.execute(
                f"""
                DESCRIBE
                SELECT *
                FROM read_parquet('{sql_path}')
                """
            ).fetchall()

            column_count = len(description)

            summary_rows.append(
                {
                    "file": parquet_path.name,
                    "size_bytes": size_bytes,
                    "size": human_size(size_bytes),
                    "rows": row_count,
                    "columns": column_count,
                }
            )

            report.append("=" * 80)
            report.append(f"FILE: {parquet_path.name}")
            report.append(f"PATH: {parquet_path}")
            report.append(f"SIZE: {human_size(size_bytes)}")
            report.append(f"ROWS: {row_count:,}")
            report.append(f"COLUMNS: {column_count}")
            report.append("")
            report.append("SCHEMA:")

            for column in description:
                column_name = column[0]
                column_type = column[1]

                report.append(
                    f"  {column_name:<40} {column_type}"
                )

            report.append("")

        except Exception as exc:

            report.append("=" * 80)
            report.append(f"ERROR READING: {parquet_path}")
            report.append(str(exc))
            report.append("")

    report.append("=" * 80)
    report.append("OTHER IMPORTANT FILES")
    report.append("")

    important_files = [
        "README.md",
        "DATA_DICTIONARY.md",
        "schema.sql",
        "specs.jsonl.gz",
        "specmine-mysql.sql.gz",
        "LICENSE.txt",
    ]

    for filename in important_files:

        matches = list(dataset_root.rglob(filename))

        if matches:

            for match in matches:
                report.append(
                    f"{filename}: "
                    f"{human_size(match.stat().st_size)} | "
                    f"{match}"
                )

        else:
            report.append(f"{filename}: NOT FOUND")

    report_text = "\n".join(report)

    report_path.write_text(
        report_text,
        encoding="utf-8",
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file",
                "size_bytes",
                "size",
                "rows",
                "columns",
            ],
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print("Dataset inventory complete.")
    print()

    print(
        f"{'TABLE':45} "
        f"{'ROWS':>15} "
        f"{'COLUMNS':>10} "
        f"{'SIZE':>12}"
    )

    print("-" * 86)

    for row in summary_rows:

        print(
            f"{row['file']:<45} "
            f"{row['rows']:>15,} "
            f"{row['columns']:>10} "
            f"{row['size']:>12}"
        )

    print()
    print(f"Detailed report: {report_path.resolve()}")
    print(f"CSV summary:     {csv_path.resolve()}")


if __name__ == "__main__":
    main()