"""Create concise EDA summaries from the public listings CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.features import _area_sqft, _map_unique, parse_amount, parse_price  # noqa: E402


def run(data_path: Path) -> None:
    frame = pd.read_csv(
        data_path,
        usecols=[
            "location",
            "Amount(in rupees)",
            "Price (in rupees)",
            "Carpet Area",
            "Transaction",
            "Furnishing",
        ],
        low_memory=False,
    )
    frame["price_rupees"] = _map_unique(frame["Amount(in rupees)"], parse_amount)
    frame["carpet_area_sqft"] = _map_unique(frame["Carpet Area"], _area_sqft)
    frame["price_per_sqft"] = _map_unique(frame["Price (in rupees)"], parse_price)
    derived_rate = frame["price_rupees"] / frame["carpet_area_sqft"]
    frame["price_per_sqft"] = frame["price_per_sqft"].where(
        frame["price_per_sqft"].gt(0), derived_rate
    )
    usable = frame.loc[
        frame["price_rupees"].gt(0) & frame["carpet_area_sqft"].gt(0)
    ].copy()

    location_summary = (
        usable.groupby("location", dropna=False)
        .agg(
            listings=("price_rupees", "size"),
            median_price_rupees=("price_rupees", "median"),
            median_price_per_sqft=("price_per_sqft", "median"),
            median_carpet_area_sqft=("carpet_area_sqft", "median"),
        )
        .sort_values("median_price_rupees", ascending=False)
        .reset_index()
    )
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    location_summary.to_csv(reports / "location_summary.csv", index=False)

    report = f"""# EDA summary

Source file: {data_path.name}

- Rows in source: {len(frame):,}
- Rows with positive price and parsed carpet area: {len(usable):,}
- Distinct locations: {usable['location'].nunique(dropna=True):,}
- Missing price values: {frame['price_rupees'].isna().sum():,}
- Missing carpet-area values: {frame['carpet_area_sqft'].isna().sum():,}
- Median listing price: INR {usable['price_rupees'].median():,.0f}
- Median carpet area: {usable['carpet_area_sqft'].median():,.0f} sq ft
- Median listing price per carpet-area sq ft: INR {usable['price_per_sqft'].median():,.0f}

The location-level table is saved to location_summary.csv. Listings are asking prices,
not a verified transaction-price series.
"""
    (reports / "eda_summary.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved {reports / 'location_summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "India_Property_Listings.csv")
    args = parser.parse_args()
    run(args.data.resolve())


if __name__ == "__main__":
    main()
