"""Feature parsing shared by training and the Streamlit application."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "BHK",
    "Carpet_Area_SqFt",
    "Super_Area_SqFt",
    "Bathrooms",
    "Balcony",
    "Floor_Number",
    "Total_Floors",
    "Parking_Spaces",
]

CATEGORICAL_FEATURES = [
    "location",
    "Status",
    "Transaction",
    "Furnishing",
    "facing",
    "overlooking",
    "Ownership",
    "property_type",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "Amount(in rupees)"


def _map_unique(series: pd.Series, mapper) -> pd.Series:
    """Map distinct values when useful, then restore the original row order."""
    if len(series) < 128:
        return series.map(mapper)

    codes, unique_values = pd.factorize(series, sort=False)
    if len(unique_values) > 0.8 * len(series):
        return series.map(mapper)
    if len(unique_values) == 0:
        return series.map(mapper)

    mapped = pd.Series(unique_values).map(mapper).to_numpy()
    missing = codes < 0
    if missing.any() and mapped.dtype.kind in "biuUS":
        mapped = mapped.astype(object)
    result = np.empty(len(codes), dtype=mapped.dtype)
    if missing.any():
        missing_values = np.empty(int(missing.sum()), dtype=object)
        missing_values.fill(mapper(np.nan))
        result[missing] = missing_values
    result[~missing] = mapped[codes[~missing]]
    return pd.Series(result, index=series.index, name=series.name)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _number(value: Any) -> float:
    """Read the first number from common listing strings."""
    if _is_missing(value):
        return np.nan
    text = str(value).strip().lower().replace(",", "")
    if not text or text in {"nan", "none", "null", "-", "not available"}:
        return np.nan
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else np.nan


def _area_sqft(value: Any) -> float:
    """Parse an area and normalize recognized units to square feet."""
    if _is_missing(value):
        return np.nan
    text = str(value).strip().lower().replace(",", "")
    amount = _number(text)
    if not np.isfinite(amount):
        return np.nan
    normalized = re.sub(r"[\s.]", "", text)
    if any(unit in normalized for unit in ("sqm", "sqmeter", "squaremeter", "m²")):
        return amount * 10.7639
    if any(unit in normalized for unit in ("sqyd", "sqyard", "squareyard", "yd²")):
        return amount * 9.0
    if "acre" in normalized:
        return amount * 43560.0
    if "hectare" in normalized or "ha" == normalized[-2:]:
        return amount * 107639.0
    # The source dataset describes carpet area in sqft; bare values use sqft.
    return amount


def _floor_values(value: Any) -> tuple[float, float]:
    if _is_missing(value):
        return np.nan, np.nan
    text = str(value).strip().lower().replace(",", "")
    if not text or text in {"nan", "none", "null", "-"}:
        return np.nan, np.nan
    match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*(?:st|nd|rd|th)?\s*(?:out of|/)\s*(\d+(?:\.\d+)?)",
        text,
    )
    if match:
        return float(match.group(1)), float(match.group(2))
    if "ground" in text or "grnd" in text:
        total_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:floor|floors)?", text)
        return 0.0, float(total_match.group(1)) if total_match else np.nan
    return _number(text), np.nan


def _parking_spaces(value: Any) -> float:
    if _is_missing(value):
        return np.nan
    text = str(value).strip().lower()
    if any(token in text for token in ("no parking", "not available", "none")):
        return 0.0
    values = re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))
    if values:
        return float(sum(float(item) for item in values))
    return np.nan


def _bhk_from_title(value: Any) -> float:
    if _is_missing(value):
        return np.nan
    match = re.search(r"(\d+(?:\.\d+)?)\s*bhk", str(value), flags=re.IGNORECASE)
    return float(match.group(1)) if match else np.nan


def _property_type(value: Any) -> str:
    if _is_missing(value):
        return "Unknown"
    text = str(value).lower()
    for label, pattern in (
        ("Plot", r"\b(plot|land)\b"),
        ("Villa", r"\bvilla\b"),
        ("Independent House", r"\b(independent house|bungalow|row house)\b"),
        ("Builder Floor", r"\b(builder floor|builder-floor)\b"),
        ("Apartment", r"\b(flat|apartment)\b"),
    ):
        if re.search(pattern, text):
            return label
    return "Other"


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the model's stable feature schema from raw listings or app inputs."""
    result = pd.DataFrame(index=frame.index)

    titles = frame.get("Title", pd.Series(index=frame.index, dtype="object"))
    floors = frame.get("Floor", pd.Series(index=frame.index, dtype="object"))
    floor_pairs = _map_unique(floors, _floor_values)

    result["BHK"] = _map_unique(titles, _bhk_from_title)
    result["Carpet_Area_SqFt"] = _map_unique(
        frame.get("Carpet Area", pd.Series(index=frame.index, dtype="object")), _area_sqft
    )
    result["Super_Area_SqFt"] = _map_unique(
        frame.get("Super Area", pd.Series(index=frame.index, dtype="object")), _area_sqft
    )
    result["Bathrooms"] = _map_unique(
        frame.get("Bathroom", pd.Series(index=frame.index, dtype="object")), _number
    )
    result["Balcony"] = _map_unique(
        frame.get("Balcony", pd.Series(index=frame.index, dtype="object")), _number
    )
    result["Floor_Number"] = floor_pairs.map(lambda pair: pair[0])
    result["Total_Floors"] = floor_pairs.map(lambda pair: pair[1])
    result["Parking_Spaces"] = _map_unique(
        frame.get("Car Parking", pd.Series(index=frame.index, dtype="object")),
        _parking_spaces,
    )

    result["location"] = _map_unique(
        frame.get("location", pd.Series(index=frame.index, dtype="object")), _category
    )
    result["Status"] = _map_unique(
        frame.get("Status", pd.Series(index=frame.index, dtype="object")), _category
    )
    result["Transaction"] = _map_unique(
        frame.get("Transaction", pd.Series(index=frame.index, dtype="object")), _category
    )
    result["Furnishing"] = _map_unique(
        frame.get("Furnishing", pd.Series(index=frame.index, dtype="object")), _category
    )
    result["facing"] = _map_unique(
        frame.get("facing", pd.Series(index=frame.index, dtype="object")), _category
    )
    result["overlooking"] = _map_unique(
        frame.get("overlooking", pd.Series(index=frame.index, dtype="object")), _category
    )
    result["Ownership"] = _map_unique(
        frame.get("Ownership", pd.Series(index=frame.index, dtype="object")), _category
    )
    result["property_type"] = _map_unique(titles, _property_type)

    return result[FEATURE_COLUMNS]


def _category(value: Any) -> str:
    if _is_missing(value):
        return np.nan
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "-", "not available"}:
        return np.nan
    return text.lower()


def parse_price(value: Any) -> float:
    """Parse the dataset's numeric per-square-foot price column."""
    if _is_missing(value):
        return np.nan
    text = str(value).strip().replace(",", "")
    return _number(text)


def parse_amount(value: Any) -> float:
    """Convert listing amounts such as '42 Lac' or '1.40 Cr' to rupees."""
    if _is_missing(value):
        return np.nan
    text = str(value).strip().lower().replace(",", "")
    if not text or any(token in text for token in ("call for price", "price on request")):
        return np.nan

    amount_units = {
        "crores": 10_000_000,
        "crore": 10_000_000,
        "cr": 10_000_000,
        "lakhs": 100_000,
        "lakh": 100_000,
        "lacs": 100_000,
        "lac": 100_000,
    }
    matches = list(
        re.finditer(
            r"(\d+(?:\.\d+)?)\s*(crores|crore|cr|lakhs|lakh|lacs|lac)\b",
            text,
        )
    )
    if matches:
        return float(
            sum(float(match.group(1)) * amount_units[match.group(2)] for match in matches)
        )
    return _number(text)
