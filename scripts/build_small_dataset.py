"""Build a deterministic small-city TSP dataset.

The source Kaggle/TSPLIB-style CSV contains many coordinate columns.  This
script keeps the first ``city_count`` city coordinates from each usable row so
that exhaustive baseline algorithms can be compared on tractable instances.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_SOURCE = Path("tsp_instances_dataset.csv")
DEFAULT_OUTPUT = Path("tsp_small_instances.csv")


def _city_columns(city_count: int) -> list[str]:
    columns: list[str] = []
    for idx in range(1, city_count + 1):
        columns.extend((f"City_{idx}_X", f"City_{idx}_Y"))
    return columns


def _is_city_column(column: str) -> bool:
    parts = column.split("_")
    return (
        len(parts) == 3
        and parts[0] == "City"
        and parts[1].isdigit()
        and parts[2] in {"X", "Y"}
    )


def _metadata_columns(columns: Iterable[str]) -> list[str]:
    return [
        column
        for column in columns
        if column not in {"TSP_Instance", "Num_Cities"} and not _is_city_column(column)
    ]


def build_small_dataset(
    source_path: str | Path = DEFAULT_SOURCE,
    output_path: str | Path = DEFAULT_OUTPUT,
    city_count: int = 10,
) -> int:
    """Write a reduced dataset and return the number of rows written.

    Rows are included only when the declared number of cities and the first
    ``city_count`` coordinate pairs are complete and numeric.
    """

    if city_count < 1:
        raise ValueError("city_count must be positive")

    source_path = Path(source_path)
    output_path = Path(output_path)
    source = pd.read_csv(source_path)

    required = ["TSP_Instance", "Num_Cities", *_city_columns(city_count)]
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(f"source CSV is missing required columns: {missing}")

    metadata = _metadata_columns(source.columns)
    output_rows: list[dict] = []

    for _, row in source.iterrows():
        try:
            declared_cities = int(row["Num_Cities"])
        except (TypeError, ValueError):
            continue
        if declared_cities < city_count:
            continue

        output_row = {
            "TSP_Instance": f"{row['TSP_Instance']}_small",
            "Num_Cities": city_count,
        }
        for column in metadata:
            output_row[column] = row[column]

        valid = True
        for column in _city_columns(city_count):
            value = pd.to_numeric(row[column], errors="coerce")
            if pd.isna(value):
                valid = False
                break
            output_row[column] = float(value)

        if valid:
            output_rows.append(output_row)

    output_columns = ["TSP_Instance", "Num_Cities", *metadata, *_city_columns(city_count)]
    output = pd.DataFrame(output_rows, columns=output_columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return len(output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="source TSP CSV path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="output small CSV path")
    parser.add_argument("--city-count", type=int, default=10, help="number of prefix cities to keep")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    count = build_small_dataset(args.source, args.output, city_count=args.city_count)
    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
