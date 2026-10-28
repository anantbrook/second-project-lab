"""Streamlit application for property price and investment-proxy insights."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.features import (
    FEATURE_COLUMNS,
    _area_sqft,
    parse_amount,
    parse_price,
    prepare_features,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "India_Property_Listings.csv"
MODELS = ROOT / "models"
REGRESSOR_PATH = MODELS / "price_regressor.joblib"
CLASSIFIER_PATH = MODELS / "investment_proxy_classifier.joblib"
BENCHMARK_PATH = MODELS / "benchmark_thresholds.json"

ASSESSMENT_COLUMNS = [
    "location",
    "Status",
    "Transaction",
    "Furnishing",
    "facing",
    "overlooking",
    "Ownership",
]
MARKET_COLUMNS = ["location", "Amount(in rupees)", "Price (in rupees)", "Carpet Area"]
CHOICE_FALLBACKS = {
    "location": ["unknown"],
    "Status": ["Ready to Move"],
    "Transaction": ["Resale"],
    "Furnishing": ["Unfurnished"],
    "facing": ["Unknown"],
    "overlooking": ["Unknown"],
    "Ownership": ["Unknown"],
}
HISTOGRAM_SAMPLE_SIZE = 50_000

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
)


def _choices(series: pd.Series, fallback: list[str]) -> list[str]:
    values = sorted(
        {
            str(value).strip()
            for value in series.dropna().unique()
            if str(value).strip() and str(value).lower() not in {"nan", "none", "null"}
        },
        key=str.casefold,
    )
    return values or fallback


@st.cache_data(show_spinner="Loading assessment options...")
def load_assessment_options(path: str, modified_ns: int) -> dict[str, list[str]]:
    frame = pd.read_csv(path, usecols=ASSESSMENT_COLUMNS, low_memory=False)
    return {
        column: _choices(frame[column], CHOICE_FALLBACKS[column])
        for column in ASSESSMENT_COLUMNS
    }


@st.cache_data(show_spinner="Summarizing market listings...")
def load_market_data(
    path: str, modified_ns: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | float]]:
    frame = pd.read_csv(path, usecols=MARKET_COLUMNS, low_memory=False)
    frame["price_rupees"] = frame["Amount(in rupees)"].map(parse_amount)
    frame["carpet_area_sqft"] = frame["Carpet Area"].map(_area_sqft)
    frame["price_per_sqft"] = frame["Price (in rupees)"].map(parse_price)
    derived_rate = frame["price_rupees"] / frame["carpet_area_sqft"]
    frame["price_per_sqft"] = frame["price_per_sqft"].where(
        frame["price_per_sqft"].gt(0), derived_rate
    )
    usable = frame.loc[
        frame["price_rupees"].gt(0) & frame["carpet_area_sqft"].gt(0)
    ]
    if usable.empty:
        empty_summary = pd.DataFrame(
            columns=["location", "median_price", "median_rate", "listings"]
        )
        empty_histogram = pd.DataFrame(columns=["price_rupees"])
        return empty_summary, empty_histogram, {
            "listing_count": 0,
            "location_count": 0,
            "median_price": 0,
        }

    grouped = (
        usable.groupby("location", dropna=False)
        .agg(
            median_price=("price_rupees", "median"),
            median_rate=("price_per_sqft", "median"),
            listings=("price_rupees", "size"),
        )
        .reset_index()
    )
    histogram = usable[["price_rupees"]]
    if len(histogram) > HISTOGRAM_SAMPLE_SIZE:
        histogram = histogram.sample(n=HISTOGRAM_SAMPLE_SIZE, random_state=42)
    summary = {
        "listing_count": int(len(usable)),
        "location_count": int(usable["location"].nunique()),
        "median_price": float(usable["price_rupees"].median()),
    }
    return grouped, histogram, summary


def _model_artifact_signature() -> tuple[int, ...]:
    paths = (REGRESSOR_PATH, CLASSIFIER_PATH, BENCHMARK_PATH)
    return tuple(path.stat().st_mtime_ns if path.exists() else -1 for path in paths)


@st.cache_resource
def load_models(artifact_signature: tuple[int, ...]):
    # The signature is part of the cache key, so retraining refreshes loaded models.
    if not all(path.exists() for path in (REGRESSOR_PATH, CLASSIFIER_PATH, BENCHMARK_PATH)):
        return None
    regressor = joblib.load(REGRESSOR_PATH)
    classifier = joblib.load(CLASSIFIER_PATH)
    benchmarks = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    return regressor, classifier, benchmarks


def _build_listing_input(
    location,
    bhk,
    property_type,
    carpet_area,
    super_area,
    bathrooms,
    balconies,
    floor_number,
    total_floors,
    parking_spaces,
    status,
    transaction,
    furnishing,
    facing,
    overlooking,
    ownership,
) -> pd.DataFrame:
    title = f"{bhk} BHK {property_type}"
    row = {
        "Title": title,
        "location": location,
        "Carpet Area": f"{carpet_area} sqft",
        "Super Area": f"{super_area} sqft" if super_area else None,
        "Bathroom": bathrooms,
        "Balcony": balconies,
        "Floor": f"{floor_number} out of {total_floors}",
        "Car Parking": f"{parking_spaces} Covered" if parking_spaces else "No parking",
        "Status": status,
        "Transaction": transaction,
        "Furnishing": furnishing,
        "facing": facing,
        "overlooking": overlooking,
        "Ownership": ownership,
    }
    return pd.DataFrame([row])


def _format_rupees(value: float) -> str:
    return f"₹{value:,.0f}"


def _recommendation_tab(options: dict[str, list[str]], models) -> None:
    regressor, classifier, benchmarks = models
    st.subheader("Property assessment")
    st.caption(
        "The recommendation compares the entered asking price per sq ft with the "
        "training-data median for the selected location."
    )

    location_values = options["location"]
    with st.form("property_form"):
        left, middle, right = st.columns(3)
        with left:
            location = st.selectbox("Location", location_values)
            property_type = st.selectbox(
                "Property type",
                ["Apartment", "Independent House", "Villa", "Builder Floor", "Plot"],
            )
            bhk = st.number_input("BHK", min_value=0, max_value=12, value=2, step=1)
            carpet_area = st.number_input(
                "Carpet area (sq ft)", min_value=1, max_value=100000, value=800, step=50
            )
            super_area = st.number_input(
                "Super area (sq ft)", min_value=0, max_value=150000, value=0, step=50
            )
        with middle:
            asking_price = st.number_input(
                "Asking price (₹)", min_value=0, max_value=2_000_000_000,
                value=5_000_000, step=100_000,
            )
            bathrooms = st.number_input("Bathrooms", min_value=0, max_value=20, value=2, step=1)
            balconies = st.number_input("Balconies", min_value=0, max_value=20, value=1, step=1)
            total_floors = st.number_input(
                "Total floors", min_value=1, max_value=200, value=10, step=1
            )
            floor_number = st.number_input(
                "Floor number", min_value=0, max_value=200, value=3, step=1
            )
        with right:
            status = st.selectbox("Status", options["Status"])
            transaction = st.selectbox(
                "Transaction", options["Transaction"]
            )
            furnishing = st.selectbox(
                "Furnishing", options["Furnishing"]
            )
            facing = st.selectbox("Facing", options["facing"])
            overlooking = st.selectbox(
                "Overlooking", options["overlooking"]
            )
            ownership = st.selectbox(
                "Ownership", options["Ownership"]
            )
            parking_spaces = st.number_input(
                "Parking spaces", min_value=0, max_value=20, value=1, step=1
            )
        annual_growth = st.slider(
            "Assumed annual price growth for the scenario",
            min_value=0.0,
            max_value=20.0,
            value=8.0,
            step=0.5,
            format="%.1f%%",
        )
        years = st.slider("Forecast horizon (years)", 1, 10, 5)
        submitted = st.form_submit_button("Assess property", type="primary")

    if not submitted:
        st.info("Enter the listing details and select **Assess property**.")
        return

    raw_input = _build_listing_input(
        location,
        bhk,
        property_type,
        carpet_area,
        super_area,
        bathrooms,
        balconies,
        floor_number,
        total_floors,
        parking_spaces,
        status,
        transaction,
        furnishing,
        facing,
        overlooking,
        ownership,
    )
    features = prepare_features(raw_input).reindex(columns=FEATURE_COLUMNS)
    predicted_price = float(np.expm1(regressor.predict(features)[0]))
    probability = float(classifier.predict_proba(features)[0, 1])
    area_price = asking_price / carpet_area
    location_key = str(location).strip().lower()
    local_benchmark = benchmarks["by_location_rupees_per_sqft"].get(
        location_key, benchmarks["fallback_rupees_per_sqft"]
    )
    below_benchmark = area_price <= float(local_benchmark)
    forecast_base = asking_price if asking_price > 0 else predicted_price
    forecast = forecast_base * (1 + annual_growth / 100) ** years

    st.divider()
    result_left, result_mid, result_right = st.columns(3)
    with result_left:
        st.metric("Estimated current listing price", _format_rupees(predicted_price))
    with result_mid:
        st.metric("Five-year scenario", _format_rupees(forecast))
    with result_right:
        st.metric("Price per carpet-area sq ft", _format_rupees(area_price))

    if below_benchmark:
        st.success(
            f"Good Investment proxy: asking price per sq ft is below the local "
            f"training median of {_format_rupees(float(local_benchmark))}."
        )
    else:
        st.warning(
            f"Review price: asking price per sq ft is above the local training "
            f"median of {_format_rupees(float(local_benchmark))}."
        )
    st.progress(probability, text=f"Classifier probability for below-median proxy: {probability:.1%}")
    st.caption(
        f"The five-year scenario compounds the {'asking price' if asking_price > 0 else 'model estimate'} "
        f"at {annual_growth:.1f}% annually for {years} years. It is not a trained future-price prediction."
    )


def _insights_tab() -> None:
    st.subheader("Market insights from the supplied listings")
    grouped, histogram, summary = load_market_data(
        str(DATA_PATH), DATA_PATH.stat().st_mtime_ns
    )
    if summary["listing_count"] == 0:
        st.warning("No usable price and area records were found.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Listings", f"{summary['listing_count']:,}")
    c2.metric("Locations", f"{summary['location_count']:,}")
    c3.metric("Median asking price", _format_rupees(float(summary["median_price"])))

    top = grouped.sort_values("median_price", ascending=False).head(15)
    fig = px.bar(
        top.sort_values("median_price"),
        x="median_price",
        y="location",
        color="listings",
        orientation="h",
        labels={"median_price": "Median asking price (₹)", "location": "Location"},
        title="15 locations with the highest median listing price",
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.histogram(
        histogram,
        x="price_rupees",
        nbins=60,
        log_x=True,
        labels={"price_rupees": "Asking price (₹)"},
        title="Listing price distribution (log scale)",
    )
    st.caption(
        f"Histogram uses a fixed-seed sample of up to {HISTOGRAM_SAMPLE_SIZE:,} usable listings."
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(
        grouped.sort_values("median_rate", ascending=False).rename(
            columns={
                "location": "Location",
                "median_price": "Median asking price (₹)",
                "median_rate": "Median price per sq ft (₹)",
                "listings": "Listing count",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def _business_use_cases_tab() -> None:
    st.subheader("Decision support and business workflows")
    st.caption(
        "The prototype can make first-pass reviews more consistent and help teams "
        "decide where to spend human attention. Measure the effect against your current workflow."
    )
    st.markdown("### A practical decision loop")
    st.markdown(
        "1. **Compare consistently:** use price per carpet-area sq ft and the location median.\n"
        "2. **Prioritize review:** use the estimate and below-median proxy to decide which listings need closer checks.\n"
        "3. **Make assumptions visible:** adjust the growth scenario instead of treating one forecast number as certain.\n"
        "4. **Verify before acting:** confirm with recent closed sales, property condition, and local expertise."
    )
    cases = [
        {
            "title": "Buyer and agent shortlists",
            "decision": "Which inquiries deserve a closer comparable-property review?",
            "support": (
                "Compare the entered asking price per carpet-area sq ft with the "
                "location median and inspect the model's current-price estimate."
            ),
            "improvement": (
                "A shared per-area comparison can reduce manual arithmetic and make "
                "first-pass reviews more consistent across agents."
            ),
            "next": "Confirm against recent closed sales, property condition, and exact micro-location.",
        },
        {
            "title": "Marketplace listing review",
            "decision": "Which seller-submitted prices should an operations team review?",
            "support": (
                "Use the estimate and local price benchmark as a manual review signal "
                "before publishing or contacting a seller."
            ),
            "improvement": (
                "A common reference can surface unusual submissions earlier and help "
                "reviewers apply the same checks; keep a human override."
            ),
            "next": "Keep the seller in control; do not automatically reject or set a listing price.",
        },
        {
            "title": "Brokerage sourcing queue",
            "decision": "Which listings should enter a human due-diligence queue first?",
            "support": (
                "The below-local-median proxy can help screen candidates for follow-up; "
                "the current app assesses one property at a time."
            ),
            "improvement": (
                "A repeatable first-pass screen can help analysts spend due-diligence "
                "time on candidates that warrant deeper review."
            ),
            "next": "For a real sourcing workflow, add batch screening plus rent, fees, title, and liquidity data.",
        },
        {
            "title": "Regional market planning",
            "decision": "Where should a team focus local research or marketing coverage?",
            "support": (
                "Market insights summarize listing counts and asking-price levels by location."
            ),
            "improvement": (
                "Combining listing volume with price levels can give teams a consistent "
                "starting point for allocating local research effort."
            ),
            "next": "Combine with dated closed transactions, inventory, demand, and geospatial data.",
        },
    ]
    for start in range(0, len(cases), 2):
        columns = st.columns(2)
        for column, case in zip(columns, cases[start:start + 2]):
            with column:
                with st.container():
                    st.markdown(f"#### {case['title']}")
                    st.markdown(f"**Decision:** {case['decision']}")
                    st.markdown(f"**What the prototype provides:** {case['support']}")
                    st.markdown(f"**How it can help:** {case['improvement']}")
                    st.markdown(f"**Production follow-up:** {case['next']}")
    st.info(
        "These are decision-support hypotheses, not proven business gains. The below-median "
        "result is a listing-price screen, not proof of profitability; the five-year value is "
        "an adjustable scenario, not a trained forecast."
    )
    with st.expander("How to measure decision improvement"):
        st.markdown(
            "Compare the assisted workflow with the team's current baseline. Track review "
            "time per listing, consistency between reviewers, and the share of flagged cases "
            "that become verified comparables or qualified leads. Measure price error only "
            "against dated, verified sale prices, split by location and property type."
        )


def _model_tab() -> None:
    st.subheader("Model performance and method")
    metrics_path = MODELS / "metrics.json"
    importance_path = MODELS / "feature_importance.json"
    if not metrics_path.exists():
        st.info("Run python src/train.py to train models and populate this page.")
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    st.write(
        f"Training used {metrics['rows_used']:,} listings "
        f"({metrics['train_rows']:,} train and {metrics['test_rows']:,} test)."
    )
    reg = metrics["regression"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Regression MAE", _format_rupees(reg["mae_rupees"]))
    c2.metric("Regression RMSE", _format_rupees(reg["rmse_rupees"]))
    c3.metric("Regression R²", f"{reg['r2']:.3f}")

    cls = metrics["classification_proxy"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Proxy F1", f"{cls['f1']:.3f}")
    c2.metric("Proxy precision", f"{cls['precision']:.3f}")
    c3.metric("Proxy recall", f"{cls['recall']:.3f}")
    st.caption(f"Classifier target: {metrics['classification_target']}.")
    if importance_path.exists():
        importance = pd.DataFrame(json.loads(importance_path.read_text(encoding="utf-8")))
        if not importance.empty:
            fig = px.bar(
                importance.head(15).sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                title="Top regression features",
            )
            st.plotly_chart(fig, use_container_width=True)
    st.code("mlflow ui --backend-store-uri ./mlruns", language="powershell")


def main() -> None:
    st.title("Real Estate Investment Advisor")
    st.caption("Listing-price estimates, local price comparisons, and five-year scenarios")
    st.warning(
        "Educational analysis only. The source contains listing prices, not verified sales or "
        "investment returns. The 'Good Investment' label is a below-local-median price proxy."
    )
    page = st.radio(
        "Dashboard section",
        ["Property assessment", "Market insights", "Business use cases", "Model details"],
        horizontal=True,
        label_visibility="collapsed",
        key="dashboard_section",
    )
    if page == "Property assessment":
        if not DATA_PATH.exists():
            st.error(f"Dataset not found: {DATA_PATH.name}. Put the CSV beside app.py.")
            st.stop()
        models = load_models(_model_artifact_signature())
        if models is None:
            st.error("Trained models are missing.")
            st.code("python src/train.py", language="powershell")
            st.stop()
        options = load_assessment_options(str(DATA_PATH), DATA_PATH.stat().st_mtime_ns)
        _recommendation_tab(options, models)
    elif page == "Market insights":
        if not DATA_PATH.exists():
            st.error(f"Dataset not found: {DATA_PATH.name}. Put the CSV beside app.py.")
            st.stop()
        _insights_tab()
    elif page == "Business use cases":
        _business_use_cases_tab()
    else:
        _model_tab()


if __name__ == "__main__":
    main()
