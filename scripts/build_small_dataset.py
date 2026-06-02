"""Build the reproducible 10-city dataset used for core search comparisons."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = "tsp_instances_dataset.csv"
DEFAULT_OUTPUT = "tsp_small_instances.csv"


def build_small_dataset(input_path: str, output_path: str, city_count: int = 10) -> int:
    """Write a deterministic prefix-city subset for every usable source row."""
    if city_count < 3:
        raise ValueError("city_count must be at least 3")

    source = pd.read_csv(input_path)
    required = ["TSP_Instance", "Num_Cities"]
    for city in range(1, city_count + 1):
        required.extend([f"City_{city}_X", f"City_{city}_Y"])

    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(f"Source dataset is missing columns: {missing}")

    usable = source[source["Num_Cities"] >= city_count].copy()
    usable = usable.dropna(subset=required)
    result = usable[required].copy()
    result["TSP_Instance"] = result["TSP_Instance"].astype(str) + "_small"
    result["Num_Cities"] = city_count

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return len(result)


def main():
    """Run the dataset builder from the command line."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Source CSV path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--cities", type=int, default=10, help="Cities per derived instance")
    args = parser.parse_args()

    rows = build_small_dataset(args.input, args.output, args.cities)
    print(f"Wrote {rows} instances to {args.output}")


if __name__ == "__main__":
    main()
